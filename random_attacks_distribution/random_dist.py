import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched
from torch.nn import BCELoss
import torch.utils.data as tud
from statistics import median, mean
import syft as sy

# hook PyTorch to PySyft, i.e. add extra functionalities to support Federated Learning and other private AI tools
hook = sy.TorchHook(torch)

# define number of clients
num_of_clients = 7

# load clients' datasets and test sets files
clients_datasets = []
booters_tests = []

for i in range(num_of_clients):
    clients_datasets.append(pd.read_csv('client' + str(i+1) + '.csv').astype('float32'))

for i in range(7):
    booters_tests.append(pd.read_csv('booter_test' + str(i+1) + '.csv').astype('float32'))

general_benign_test = pd.read_csv('general_benign_test.csv').astype('float32')

"""## Making and normalization of test sets

Make test sets to compare the performance of each client on each attack and on general benign traffic with and without federated learning.
"""

nonfed_attacks = [copy.deepcopy(booters_tests) for i in range(num_of_clients)]
nonfed_benign = [copy.copy(general_benign_test) for i in range(num_of_clients)]
fed_attacks = copy.deepcopy(booters_tests)
fed_benign = copy.copy(general_benign_test)

"""Without-FL test sets need to have the same features as the clients' datasets they correspond to."""

for client in range(num_of_clients):
    to_drop = []
    for feature in general_benign_test.columns:
        if feature not in clients_datasets[client].columns:
            to_drop.append(feature)
    for booter in range(7):
        nonfed_attacks[client][booter].drop(columns=to_drop, inplace=True)
    nonfed_benign[client].drop(columns=to_drop, inplace=True)

"""All test and train sets need to be normalized. Train sets will be normalized privately by each client. The same scalers used for the train sets, will also normalize all without-FL test sets. However, with-FL test sets need to be normalized using an aggregated scaler from all clients, for which to be calculated clients need to send their scalers to the central entity of the federated learning. Of course, for the aggregated scaler, every client only contributes to the features that has chosen after feature selection step."""

scaling_results = []
for client in range(num_of_clients):
    scaler = MinMaxScaler()
    clients_datasets[client].iloc[:, :-1] = scaler.fit_transform(clients_datasets[client].iloc[:, :-1])
    # use trainset-fitted scaler for without-FL test sets
    for booter in range(7):
        nonfed_attacks[client][booter].iloc[:, :-1] = scaler.transform(nonfed_attacks[client][booter].iloc[:, :-1])
    nonfed_benign[client].iloc[:, :-1] = scaler.transform(nonfed_benign[client].iloc[:, :-1])
    # each client sends a dictionary with keys being features names
    # and values being tuples of min and max of features 
    scaling_results.append(dict(zip(clients_datasets[client].columns[:-1], zip(scaler.data_min_, scaler.data_max_))))

"""For each feature, aggregated scaler will be calculated using the minimum and maximum values of feature that appear in the union of all clients' datasets."""

# initialize scaler and useful dictionaries
scaler = MinMaxScaler()
scaler.fit(general_benign_test.iloc[:, :-1])
scaler.min_ = []
scaler.scale_ = []
mins = {i:float('Inf') for i in general_benign_test.columns[:-1]}
maxes = {i:-float('Inf') for i in general_benign_test.columns[:-1]}

# compute appropriate min and max values of features
for client in range(num_of_clients):
    for feature, t in scaling_results[client].items():
        if t[0] < mins[feature]:
            mins[feature] = t[0]
        if t[1] > maxes[feature]:
            maxes[feature] = t[1]

# if mins[i] = Inf, feature i is dropped at all clients
# assign mins[i] = 0 and maxes[i] = 1 which lead to not scaling feature i
for feature in mins:
    if mins[feature] == float('Inf'):
        mins[feature] = 0
        maxes[feature] = 1

# pass values to the scaler
for feature in general_benign_test.columns[:-1]:
    scaler.min_.append(-mins[feature]/(maxes[feature]-mins[feature]))
    scaler.scale_.append(1/(maxes[feature]-mins[feature]))

"""Normalize with-FL test sets using above calculated aggregated scaler."""

for booter in range(7):
    fed_attacks[booter].iloc[:, :-1] = scaler.transform(fed_attacks[booter].iloc[:, :-1])
fed_benign.iloc[:, :-1] = scaler.transform(fed_benign.iloc[:, :-1])

"""**Note:** Some features of the with-FL test sets are indeed not scaled, because all clients happened to have dropped these features. The code is general, taking into account that there could be a client who would keep a feature that others have dropped. This will also be taken into account in the learning process, where the central model's weights will all be initialized to zero, so that unused features do not affect the testing performed by the central entity.

Split with-FL test sets to validation and testing sets.
"""

fed_attacks_val = [fed_attacks[b].iloc[:len(fed_attacks[b])//2, :] for b in range(7)]
fed_attacks_test = [fed_attacks[b].iloc[len(fed_attacks[b])//2:, :] for b in range(7)]

fed_benign_val = fed_benign.iloc[:len(fed_benign)//2, :]
fed_benign_test = fed_benign.iloc[len(fed_benign)//2:, :]

"""## Define training parameters and models, transform datasets to tensors, send data to clients, create dataloaders"""

# create clients
clients = [sy.VirtualWorker(hook, id='client'+str(i+1)) for i in range(num_of_clients)]

# define the args
args = {
    'use_cuda' : True,
    'batch_size' : 128,
    'test_batch_size' : 1000,
    'lr' : 0.01,
    'log_interval' : 500,
    'epochs' : 7
}

# check to use GPU or not
use_cuda = args['use_cuda'] and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# create a simple feedforward network
# n features as input, 2*n+1 hidden layer neurons, 1 output for binary classification
class MLP(nn.Module):
    
    def __init__(self, n):
        super(MLP, self).__init__()
        self.n = n
        
        self.layers = nn.Sequential(
            nn.Linear(in_features=n, out_features=2*n+1),
            nn.ReLU(),
            nn.Linear(in_features=2*n+1, out_features=1),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        return self.layers(x)

# transform to tensors
train_features = [torch.tensor(cd.iloc[:, :-1].to_numpy()) for cd in clients_datasets]
train_target = [torch.tensor(cd['target'].to_numpy()) for cd in clients_datasets]

nonfed_attacks_test_features = [[torch.tensor(nonfed_attacks[c][b].iloc[:len(nonfed_attacks[c][b])//2, :-1].to_numpy()) for b in range(7)] for c in range(num_of_clients)]
nonfed_attacks_test_target = [[torch.tensor(nonfed_attacks[c][b].iloc[:len(nonfed_attacks[c][b])//2, -1].to_numpy()) for b in range(7)] for c in range(num_of_clients)]

nonfed_attacks_val_features = [[torch.tensor(nonfed_attacks[c][b].iloc[len(nonfed_attacks[c][b])//2:, :-1].to_numpy()) for b in range(7)] for c in range(num_of_clients)]
nonfed_attacks_val_target = [[torch.tensor(nonfed_attacks[c][b].iloc[len(nonfed_attacks[c][b])//2:, -1].to_numpy()) for b in range(7)] for c in range(num_of_clients)]

nonfed_benign_test_features = [torch.tensor(nonfed_benign[c].iloc[:len(nonfed_benign[c])//2, :-1].to_numpy()) for c in range(num_of_clients)]
nonfed_benign_test_target = [torch.tensor(nonfed_benign[c].iloc[:len(nonfed_benign[c])//2, -1].to_numpy()) for c in range(num_of_clients)]

nonfed_benign_val_features = [torch.tensor(nonfed_benign[c].iloc[len(nonfed_benign[c])//2:, :-1].to_numpy()) for c in range(num_of_clients)]
nonfed_benign_val_target = [torch.tensor(nonfed_benign[c].iloc[len(nonfed_benign[c])//2:, -1].to_numpy()) for c in range(num_of_clients)]

fed_attacks_val_features = [torch.tensor(fed_attacks_val[b].iloc[:, :-1].to_numpy()) for b in range(7)]
fed_attacks_val_target = [torch.tensor(fed_attacks_val[b]['target'].to_numpy()) for b in range(7)]

fed_attacks_test_features = [torch.tensor(fed_attacks_test[b].iloc[:, :-1].to_numpy()) for b in range(7)]
fed_attacks_test_target = [torch.tensor(fed_attacks_test[b]['target'].to_numpy()) for b in range(7)]

fed_benign_val_features = torch.tensor(fed_benign_val.iloc[:, :-1].to_numpy())
fed_benign_val_target = torch.tensor(fed_benign_val['target'].to_numpy())

fed_benign_test_features = torch.tensor(fed_benign_test.iloc[:, :-1].to_numpy())
fed_benign_test_target = torch.tensor(fed_benign_test['target'].to_numpy())

# distribute data across workers
# normally there is no need to distribute data, since it is already at the clients
# this is more of a simulation of federated learning
train_datasets = [sy.BaseDataset(train_features[i].send(c), train_target[i].send(c)) for i, c in enumerate(clients)]
federated_dataset = sy.FederatedDataset(train_datasets)
federated_train_loader = sy.FederatedDataLoader(federated_dataset, batch_size=args['batch_size'], shuffle=True)

# test data remains at the central entity
nonfed_attacks_test_datasets = [[tud.TensorDataset(nonfed_attacks_test_features[c][b], nonfed_attacks_test_target[c][b]) for b in range(7)] for c in range(num_of_clients)]
nonfed_attacks_test_loaders = [[tud.DataLoader(nonfed_attacks_test_datasets[c][b], batch_size=args['test_batch_size'], shuffle=True) for b in range(7)] for c in range(num_of_clients)]

nonfed_attacks_val_datasets = [[tud.TensorDataset(nonfed_attacks_val_features[c][b], nonfed_attacks_val_target[c][b]) for b in range(7)] for c in range(num_of_clients)]
nonfed_attacks_val_loaders = [[tud.DataLoader(nonfed_attacks_val_datasets[c][b], batch_size=args['test_batch_size'], shuffle=True) for b in range(7)] for c in range(num_of_clients)]

nonfed_benign_test_datasets = [tud.TensorDataset(nonfed_benign_test_features[c], nonfed_benign_test_target[c]) for c in range(num_of_clients)]
nonfed_benign_test_loaders = [tud.DataLoader(nonfed_benign_test_datasets[c], batch_size=args['test_batch_size'], shuffle=True) for c in range(num_of_clients)]

nonfed_benign_val_datasets = [tud.TensorDataset(nonfed_benign_val_features[c], nonfed_benign_val_target[c]) for c in range(num_of_clients)]
nonfed_benign_val_loaders = [tud.DataLoader(nonfed_benign_val_datasets[c], batch_size=args['test_batch_size'], shuffle=True) for c in range(num_of_clients)]

fed_attacks_val_datasets = [tud.TensorDataset(fed_attacks_val_features[b], fed_attacks_val_target[b]) for b in range(7)]
fed_attacks_val_loaders = [tud.DataLoader(fed_attacks_val_datasets[b], batch_size=args['test_batch_size'], shuffle=True) for b in range(7)]

fed_attacks_test_datasets = [tud.TensorDataset(fed_attacks_test_features[b], fed_attacks_test_target[b]) for b in range(7)]
fed_attacks_test_loaders = [tud.DataLoader(fed_attacks_test_datasets[b], batch_size=args['test_batch_size'], shuffle=True) for b in range(7)]

fed_benign_val_dataset = tud.TensorDataset(fed_benign_val_features, fed_benign_val_target)
fed_benign_val_loader = tud.DataLoader(fed_benign_val_dataset, batch_size=args['test_batch_size'], shuffle=True)

fed_benign_test_dataset = tud.TensorDataset(fed_benign_test_features, fed_benign_test_target)
fed_benign_test_loader = tud.DataLoader(fed_benign_test_dataset, batch_size=args['test_batch_size'], shuffle=True)

"""## Train, test, aggregation, trust computation functions"""

# classic torch code for training, except for the federated part
def train(args, models, device, train_loader, optimizers, epoch, view_log=False):
    for c, m in models.items():
        m.train()
        # send models to workers
        m.send(c)

    # iterate over federated data client by client
    # of course, in reality all clients would train their models at the same time
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizers[data.location].zero_grad()
        output = models[data.location](data)

        # loss is a ptr to the tensor loss at the remote location
        loss = BCELoss()(output, target.view_as(output))
        # call backward() on the loss ptr, that will send the command to call
        # backward on the actual loss tensor present on the remote machine
        loss.backward()
        optimizers[data.location].step()

        if view_log and batch_idx % args['log_interval'] == 0:
            # get back loss, that was created at remote worker
            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\\tLoss: {:.6f}\\tWorker: {}'.format(
                    epoch, 
                    batch_idx * args['batch_size'], # number of packets done
                    len(train_loader) * args['batch_size'], # total packets
                    100. * batch_idx / len(train_loader), # percentage of batches done
                    loss,
                    data.location.id
                )
            )

    # get back models for aggregation
    for m in models.values():
        m = m.get()

# classic torch code for testing
def test(model, device, test_loader, testType='Validation', view_log=False):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add losses together
            test_loss += BCELoss(reduction='sum')(output, target.view_as(output)).item()

            # favour class 0
            output = torch.max(output-0.2, torch.zeros(size=output.shape).to(device))
            
            # get the index of the max probability class and adjust correctly classified samples
            pred = torch.round(output)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    if view_log:
        print(testType + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            accuracy))
    
    return accuracy

def aggregate(central_model, models, weights, trust, num_of_clients_in_weights):
    with torch.no_grad():
        # dataXtrust values needed for normalization later
        dataXtrust_hidden_weight = torch.zeros(size=central_model.layers[0].weight.shape).to(device)
        dataXtrust_hidden_bias = torch.zeros(size=central_model.layers[0].bias.shape).to(device)
        dataXtrust_output_weight = torch.zeros(size=central_model.layers[2].weight.shape).to(device)
        dataXtrust_output_bias = 0
        # firstly compute new aggregated weight values
        # to do so we start by taking the sum of the weights of all clients
        for i, c in enumerate(clients):
            # each client only contributes to chosen features (i.e. columns of weights arrays)
            # for each of these features (columns), the aggregation uses the first x elements (rows) of central model weights
            # where x is the number of hidden layer neurons of client and is equal to 2*(number_of_features_of_client)+1
            rows = 2*models[c].n+1
            for j, feature in enumerate(clients_datasets[i].columns[:-1]):
                # find the index of feature in the central_model
                index = general_benign_test.columns[:-1].get_loc(feature)
                weights['hidden_mean_weight'][:rows, index] += models[c].layers[0].weight.data[:, j].clone()*len(clients_datasets[i])*trust[c]
                dataXtrust_hidden_weight[:rows, index] += len(clients_datasets[i])*trust[c]
            # the rest of the weights don't have to be calculated feature-wise
            weights['hidden_mean_bias'][:rows] += models[c].layers[0].bias.data.clone()*len(clients_datasets[i])*trust[c]
            dataXtrust_hidden_bias[:rows] += len(clients_datasets[i])*trust[c]
            weights['output_mean_weight'][0, :rows] += models[c].layers[2].weight.data[0, :].clone()*len(clients_datasets[i])*trust[c]
            dataXtrust_output_weight[0, :rows] += len(clients_datasets[i])*trust[c]
            weights['output_mean_bias'] += models[c].layers[2].bias.data.clone()*len(clients_datasets[i])*trust[c]
            dataXtrust_output_bias += len(clients_datasets[i])*trust[c]

        # diminish influence of rare weights (i.e. weights of features that few clients have)
        dataXtrust_hidden_weight[num_of_clients_in_weights['hidden'] == 3] *= 2
        dataXtrust_hidden_weight[num_of_clients_in_weights['hidden'] == 2] *= 3
        dataXtrust_hidden_weight[num_of_clients_in_weights['hidden'] == 1] *= 4
        dataXtrust_output_weight[num_of_clients_in_weights['output'].reshape(1, 1, central_model.layers[2].weight.shape[1]) == 3] *= 2
        dataXtrust_output_weight[num_of_clients_in_weights['output'].reshape(1, 1, central_model.layers[2].weight.shape[1]) == 2] *= 3
        dataXtrust_output_weight[num_of_clients_in_weights['output'].reshape(1, 1, central_model.layers[2].weight.shape[1]) == 1] *= 4

        # change zero dataXtrust values to ones
        dataXtrust_hidden_weight[dataXtrust_hidden_weight == 0] = 1
        dataXtrust_hidden_bias[dataXtrust_hidden_bias == 0] = 1
        dataXtrust_output_weight[(dataXtrust_output_weight == 0).reshape(1, 1, central_model.layers[2].weight.shape[1])] = 1

        # and then we normalize the sum taking into account number of data and trust value for each client
        # again parts of weights' arrays are normalized with respect only to clients that contributed to these parts
        weights['hidden_mean_weight'] /= dataXtrust_hidden_weight
        weights['hidden_mean_bias'] /= dataXtrust_hidden_bias
        weights['output_mean_weight'] /= dataXtrust_output_weight
        weights['output_mean_bias'] /= dataXtrust_output_bias

        # secondly copy new weight values to the local models of all clients
        for i, c in enumerate(clients):
            rows = 2*models[c].n+1
            for j, feature in enumerate(clients_datasets[i].columns[:-1]):
                index = general_benign_test.columns[:-1].get_loc(feature)
                models[c].layers[0].weight.data[:, j] = weights['hidden_mean_weight'][:rows, index].clone()
            # the rest of the weights don't have to be copied feature-wise
            models[c].layers[0].bias.data = weights['hidden_mean_bias'][:rows].clone()
            models[c].layers[2].weight.data[0, :] = weights['output_mean_weight'][0, :rows].clone()
            models[c].layers[2].bias.data = weights['output_mean_bias'].clone()

        # and finally copy to the central model for the test set
        central_model.layers[0].weight.data = weights['hidden_mean_weight'].clone()
        central_model.layers[0].bias.data = weights['hidden_mean_bias'].clone()
        central_model.layers[2].weight.data = weights['output_mean_weight'].clone()
        central_model.layers[2].bias.data = weights['output_mean_bias'].clone()

def computeTrust(models, trust, r, s, num_of_clients_in_weights):
    # dev[i] shows how much the weights of model of client i differ from the models of all other clients
    # it is calculated in accordance with the relevant paper, but also taking into account the heterogeneity of models 
    dev = [0 for i in clients]
    for i, c in enumerate(clients):
        for j, cc in enumerate(clients):
            # the smallest model defines the number of weights of rows (neurons) that will be compared
            rows = min(2*models[c].n+1, 2*models[cc].n+1)
            # between 2 clients, only weights of features that both have chosen are compared
            for indexi, feature in enumerate(clients_datasets[i].columns[:-1]): 
                try:
                    # find the index of the column of feature in cc client, provided that cc has chosen this feature
                    indexj = clients_datasets[j].columns[:-1].get_loc(feature)
                except:
                    # go to the next feature, if current feature not chosen by cc
                    continue
                # for hidden layer, add to dev the sum of squared differences of weights of models divided by the number of clients which have each weight
                to_divide = num_of_clients_in_weights['hidden'][:rows, general_benign_test.columns[:-1].get_loc(feature)]
                difference = models[cc].layers[0].weight.data[:rows, indexj].cpu() - models[c].layers[0].weight.data[:rows, indexi].cpu()
                dev[i] += np.sum(difference.numpy()**2 / to_divide)
            # output layer weights don't have to be compared feature-wise
            # same as above for the output layer
            difference = models[cc].layers[2].weight.data[0, :rows].cpu() - models[c].layers[2].weight.data[0, :rows].cpu()
            dev[i] += np.sum(difference.numpy()**2 / num_of_clients_in_weights['output'][0, :rows])

    # I[i] = 1 if client i acts normally and 0 if malicious or malfunctions
    I = [1 if d <= 1.5*median(sorted(dev)) else 0 for d in dev]

    # compute new r, s values for every client
    for i in range(len(clients)):
        p1 = 0.2
        #p2 = lambda x: x/median(sorted(dev)) if x/median(sorted(dev)) > 3 and x > 30 else (x/1000 if x > 1000 else (0.01 if I[i] == 1 and s[i] > 10 else 0.7))
        p2 = lambda x: 0.8
        r[i] = p1*r[i] + I[i]
        s[i] = p2(dev[i])*s[i] + 1 - I[i]

    # compute new trust value of every client
    for i, c in enumerate(clients):
        trust[c] = (r[i]+1)/(r[i]+s[i]+2)

"""## Results without FL"""

# clients' models, optimizers and schedulers for learning rate
models = {clients[c]:MLP(len(clients_datasets[c].columns[:-1])).to(device) for c in range(num_of_clients)}
optimizers = {c:optim.SGD(models[c].parameters(), lr=args['lr']) for c in clients}
# decreasing learning rate
lamda = lambda epoch: 1 if epoch < 1 else 0.1
schedulers = {i:sched.LambdaLR(optimizers[i], lr_lambda=lamda) for i in clients}

# choose best model of all epochs (initialization)
best_models = [copy.deepcopy(models[c]) for c in models]
best_accuracies = [[0 for i in range(8)] for j in range(7)]

for epoch in range(1, args['epochs'] + 1):
    train(args, models, device, federated_train_loader, optimizers, epoch)
    for scheduler in schedulers.values():
        scheduler.step()
    for c in range(num_of_clients):
        accuracies = []
        for b in range(7):
            accuracies.append(test(models[clients[c]], device, nonfed_attacks_val_loaders[c][b]))
        accuracies.append(test(models[clients[c]], device, nonfed_benign_val_loaders[c]))

        # update best models
        mean_attacks = mean(accuracies[:4] + accuracies[5:])
        best_mean_attacks = mean(best_accuracies[c][:4] + best_accuracies[c][5:])
        # false positives are more harmful than false negatives
        if mean_attacks + 1.2*accuracies[7] > best_mean_attacks + 1.2*best_accuracies[c][7]:
            best_models[c] = copy.deepcopy(models[clients[c]])
            best_accuracies[c] = accuracies

# final testings
for c in range(num_of_clients):
    print('Client ' + str(c+1) + ' best:')
    for b in range(7):
        print('Booter ' + str(b+1) + ': ', end='')
        test(best_models[c], device, nonfed_attacks_test_loaders[c][b], testType='Test', view_log=True)
    print('Benign traffic: ', end='')
    test(best_models[c], device, nonfed_benign_test_loaders[c], testType='Test', view_log=True)

"""## FL training with 7 clients"""

# central model
central_model = MLP(len(general_benign_test.columns[:-1])).to(device)
# initialize weights of central model to zero,
# so that features which are dropped by all clients do not affect testing
central_model.layers[0].weight.data.fill_(0)
central_model.layers[0].bias.data.fill_(0)
central_model.layers[2].weight.data.fill_(0)
central_model.layers[2].bias.data.fill_(0)

# clients' models, optimizers and schedulers for learning rate
# note that central entity knows the chosen features of each client from the preprocessing procedure
models = {clients[c]:MLP(len(clients_datasets[c].columns[:-1])).to(device) for c in range(num_of_clients)}
optimizers = {c:optim.SGD(models[c].parameters(), lr=args['lr']) for c in clients}
# some clients may work better with another learning rate value
optimizers[clients[5]] = optim.SGD(models[clients[5]].parameters(), lr=0.5*args['lr'])
optimizers[clients[6]] = optim.SGD(models[clients[6]].parameters(), lr=0.5*args['lr'])
# decreasing learning rate
lamda = lambda epoch: 1 if epoch < 1 else 0.1
schedulers = {i:sched.LambdaLR(optimizers[i], lr_lambda=lamda) for i in clients}

# initialization of dictionary for models aggregation
weights = {'hidden_mean_weight' : torch.zeros(size=central_model.layers[0].weight.shape).to(device),
           'hidden_mean_bias' : torch.zeros(size=central_model.layers[0].bias.shape).to(device),
           'output_mean_weight' : torch.zeros(size=central_model.layers[2].weight.shape).to(device),
           'output_mean_bias' : torch.zeros(size=central_model.layers[2].bias.shape).to(device)}

# trust values
trust = {i:0 for i in clients}
r = [0 for i in clients]
s = [0 for i in clients]

# for each weight of central_model, count the number of clients which contain this weight in their models
# needed to compute the trust value of each client
num_of_clients_in_weights = {'hidden' : np.zeros(central_model.layers[0].weight.shape),
                             'output' : np.zeros(central_model.layers[2].weight.shape)}
for i, c in enumerate(clients):
    rows = 2*models[c].n+1
    num_of_clients_in_weights['output'][0, :rows] += 1
    for j, feature in enumerate(clients_datasets[i].columns[:-1]):
        index = general_benign_test.columns[:-1].get_loc(feature)
        num_of_clients_in_weights['hidden'][:rows, index] += 1

# choose best model of all epochs (initialization)
best_model = copy.deepcopy(central_model)
best_accuracies = [0 for i in range(8)]

for epoch in range(1, args['epochs'] + 1):
    train(args, models, device, federated_train_loader, optimizers, epoch)
    for scheduler in schedulers.values():
        scheduler.step()

    computeTrust(models, trust, r, s, num_of_clients_in_weights)
    aggregate(central_model, models, weights, trust, num_of_clients_in_weights)

    accuracies = []
    for b in range(7):
        accuracies.append(test(central_model, device, fed_attacks_val_loaders[b]))
    accuracies.append(test(central_model, device, fed_benign_val_loader))

    # update best model
    if min(accuracies[:4] + accuracies[5:]) > 75:
        mean_attacks = mean(accuracies[:4] + accuracies[5:])
        best_mean_attacks = mean(best_accuracies[:4] + best_accuracies[5:])
        # false positives are more harmful than false negatives
        if mean_attacks + 1.2*accuracies[7] > best_mean_attacks + 1.2*best_accuracies[7]:
            best_model = copy.deepcopy(central_model)
            best_accuracies = accuracies

# final testings
print('Federated Learning best:')
for b in range(7):
    print('Booter ' + str(b+1) + ': ', end='')
    test(best_model, device, fed_attacks_test_loaders[b], testType='Test', view_log=True)
print('Benign traffic: ', end='')
test(best_model, device, fed_benign_test_loader, testType='Test', view_log=True)
