import pandas as pd
from collections import Counter
import hashlib
import random
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

# load datasets
df = []
df.append(pd.read_csv("datasets/anon-Booter_dns1.pcap.csv"))
df.append(pd.read_csv("datasets/anon-Booter_dns2.pcap.csv"))
df.append(pd.read_csv("datasets/anon-Booter_dns3.pcap.csv"))
df.append(pd.read_csv("datasets/anon-Booter_dns4.pcap.csv"))
df.append(pd.read_csv("datasets/anon-Booter_dns5.pcap.csv"))
df.append(pd.read_csv("datasets/anon-Booter_dns6.pcap.csv"))
df.append(pd.read_csv("datasets/anon-Booter_dns7.pcap.csv"))
df.append(pd.read_csv("datasets/wideg_dns_cleared.csv"))
df.append(pd.read_csv("datasets/widef_dns_cleared.csv"))

"""At first, delete columns that do not exist in both datasets and, if needed, reorder the rest columns of one dataset to match the order of the other. For now, keep the *dst_AS* column of the *wide* datasets, because it will help distribute samples to clients. Also, drop *ip.src* and *ip.dst*, because they are too specific and do not really contribute to distinguishing attack and benign traffic in general case."""

for i in range(7):
    df[i].drop(columns=['ip.src', 'ip.dst', 'frame.time_relative', 'dns.id', 'dns.flags.z', 'dns.flags.rcode', 'dns.qry.class'], inplace=True)

for i in range(7, 9):
    df[i].drop(columns=['ip.src', 'ip.dst', 'src_AS'], inplace=True)

"""Drop samples with missing values."""

for i in range(9):
    df[i].dropna(axis=0, inplace=True)

"""Add a column for the target, i.e. 0 for benign traffic and 1 for attack traffic."""

for i in range(7):
    df[i] = df[i].assign(target = [1.0 for j in range(len(df[i]))])

for i in range(7, 9):
    df[i] = df[i].assign(target = [0.0 for j in range(len(df[i]))])

"""Notice that *dns.qry.name* and *dns.qry.type* are nominal categorical features. Transform the former to a numerical value via hashing. As far as the latter is concerned, one-hot encoding could be used, but that would add 255 columns to the dataset. A more efficient solution would be binary encoding. However, when trying to fit such an encoder in the concatenation of *dns.qry.type* columns of all datasets, memory issues occur (RAM gets exhausted). Another choice is to leave the feature as it is.

**Note:** Representing the values of a nominal feature as numbers is theoretically wrong in machine learning, because some order among these values is supposed, without such existing. However, due to the fact that some names or types appear in high frequence in one kind of traffic (e.g. DNS root zone name and DNS ANY type query in attack traffic), and after testing in practice, it was found that this representation tends to improve the results of the classifier, compared to dropping the aforementioned columns.
"""

for i in range(9):
    df[i]['dns.qry.name'] = df[i]['dns.qry.name'].apply(lambda a: int(hashlib.sha256(a.encode('utf-8')).hexdigest(), 16) % 10**8)

"""Convert to float numbers."""

for i in range(7):
    df[i] = df[i].astype('float32')

for i in range(7, 9):
    df[i].iloc[:, :-2] = df[i].iloc[:, :-2].astype('float32')

"""Make a general benign traffic test set with samples from both *wide* datasets to compare clients' performance in recognizing benign traffic before and after federated learning."""

# shuffle datasets
for i in range(9):
    df[i] = df[i].sample(frac=1)

# make test set
general_benign_test = pd.concat([df[7].iloc[:150000, :], df[8].iloc[:150000, :]])
# remove selected samples to avoid selecting any of them for train set
df[7] = df[7].iloc[150000:, :]
df[8] = df[8].iloc[150000:, :]
# remove dst_AS column
general_benign_test.drop(columns=['dst_AS'], inplace=True)

# shuffle test set
general_benign_test = general_benign_test.sample(frac=1).reset_index(drop=True)

"""Use the *dst_AS* feature of the *wide* datasets to split the data to clients. Each different destination AS number will represent a distinct federated client in this project. This is realistic because every packet will be received by its destination AS, which could really be a federated learning client trying to classify its receiving packets as benign or attack traffic.

Plan is to do training with 7, 14 and 21 clients. So, make 21 clients using the most frequent destination AS numbers in both *wide* datasets. In each case, about half of the clients will come from the *wideg* dataset and half from the *widef* dataset.
"""

# not all values of dst_AS columns are of the same type 
for i in range(7, 9):
    df[i].iloc[:, -2] = df[i].iloc[:, -2].astype('str')

# show the most frequent AS numbers of wideg dataset
gDstAS = Counter(df[7]['dst_AS'])
gDstAS = sorted(gDstAS.items(), key=lambda x: x[1], reverse=True)

# show the most frequent AS numbers of widef dataset
fDstAS = Counter(df[8]['dst_AS'])
fDstAS = sorted(fDstAS.items(), key=lambda x: x[1], reverse=True)

# see if some of the most frequent AS numbers have packets in both datasets
totalDstAS = Counter(pd.concat([df[7]['dst_AS'], df[8]['dst_AS']]))
totalDstAS = sorted(totalDstAS.items(), key=lambda x: x[1], reverse=True)

# take some of the most frequent AS numbers of wideg and widef datasets as clients,
# but try to have about the same number of packets from the two datasets,
# and assign benign traffic to clients' datasets
# each client takes all packets headed at it (i.e. packets from both datasets)

clients_datasets = []

# leave out AS's with too many packets for time efficiency
gDstAS = gDstAS[10:]
fDstAS = fDstAS[3:]

# avoid choosing packets with unknown AS destination
for l in [gDstAS, fDstAS]:
    for t in l:
        if t[0] == 'Unknown' or t[0] == '16509':
            l.remove((t[0], t[1]))

# avoid choosing same AS twice
chosen_ASs = []

for i in range(7):
    if i % 2 == 0:
        AS_tuple = gDstAS[i//2]
        while AS_tuple[0] in chosen_ASs:
            gDstAS.remove((AS_tuple[0], AS_tuple[1]))
            AS_tuple = gDstAS[i//2]
        clients_datasets.append(pd.concat([df[7][df[7]['dst_AS'] == AS_tuple[0]], df[8][df[8]['dst_AS'] == AS_tuple[0]]]))
        chosen_ASs.append(AS_tuple[0])        
    else:
        AS_tuple = fDstAS[i//2]
        while AS_tuple[0] in chosen_ASs:
            fDstAS.remove((AS_tuple[0], AS_tuple[1]))
            AS_tuple = fDstAS[i//2]
        clients_datasets.append(pd.concat([df[7][df[7]['dst_AS'] == AS_tuple[0]], df[8][df[8]['dst_AS'] == AS_tuple[0]]]))
        chosen_ASs.append(AS_tuple[0])

"""Check the results of the splitting."""

random.Random(8).shuffle(clients_datasets)

"""Randomly Distribute attack traffic to clients."""

random_distribution = list(range(7))
random.shuffle(random_distribution)
for i, j in zip(range(7), random_distribution):
    clients_datasets[i] = pd.concat([clients_datasets[i], df[j%7].iloc[:len(clients_datasets[i]), :]], ignore_index=True)
    # remove used samples from booter datasets
    df[j%7] = df[j%7].iloc[len(clients_datasets[i]):, :]
    # shuffle client dataset
    clients_datasets[i] = clients_datasets[i].sample(frac=1)
    print('Client ' + str(i+1) + ':')
    print('\tNumber of packets: ' + str(len(clients_datasets[i])) + '\tAS number: ' + str(clients_datasets[i].iloc[0, -2]) + '\tBooter' + str(j+1))

"""Drop the *dst_AS* column."""

for i in range(7):
    clients_datasets[i].drop(columns=['dst_AS'], inplace=True)

"""Clients' performance in recognizing all *booter* attacks will be tested before and after federated learning process. So, keep some samples of each *booter* dataset."""

booters_tests = [df[i].iloc[:50000, :] for i in range(7)]

"""Perform a feature selection technique (random forest) to further reduce the feature set for efficient training. Each client runs random forest privately on its own data multiple times and calculates the average importance score for each feature."""

rfc = RandomForestClassifier()
importance = [[] for i in clients_datasets]
for i in range(7):
    importance[i] = [0 for j in clients_datasets[i].columns[:-1]]
    # run random forest 10 times
    for n in range(10):
        rfc.fit(clients_datasets[i].iloc[:, :-1], clients_datasets[i].iloc[:, -1])
        importance[i] += rfc.feature_importances_

    importance[i] /= 10

"""Clients drop the features that they found less important. Each client drops features with importance score less than 0.01."""

for i in range(7):
    initial_columns = clients_datasets[i].columns
    for j in range(len(importance[i])):
        if importance[i][j] < 0.01:
            clients_datasets[i].drop(columns=initial_columns[j], inplace=True)

"""Before training can start, datasets need to be normalized appropriately. For now, save all the necessary dataframes."""

for i in range(7):
    clients_datasets[i].to_csv('client' + str(i+1) + '.csv', index=False)

general_benign_test.to_csv('general_benign_test.csv', index=False)

for i in range(7):
    booters_tests[i].to_csv('booter_test' + str(i+1) + '.csv', index=False)

print('PREPROCESSING DONE!')
