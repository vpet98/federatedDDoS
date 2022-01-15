# for analysis of below script look at file demo_datasets_preprocessing.ipynb

import pandas as pd
import hashlib
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

# load datasets
df1 = pd.read_csv("datasets/anon-Booter_dns1.pcap.csv")
df2 = pd.read_csv("datasets/anon-Booter_dns2.pcap.csv")
df3 = pd.read_csv("datasets/anon-Booter_dns3.pcap.csv")
df4 = pd.read_csv("datasets/anon-Booter_dns4.pcap.csv")
df5 = pd.read_csv("datasets/anon-Booter_dns5.pcap.csv")
df6 = pd.read_csv("datasets/anon-Booter_dns6.pcap.csv")
df7 = pd.read_csv("datasets/anon-Booter_dns7.pcap.csv")
dfg = pd.read_csv("datasets/wideg_dns_cleared.csv")
dff = pd.read_csv("datasets/widef_dns_cleared.csv")

print("Datasets shapes:")
print("\tdf1: " + str(df1.shape))
print("\tdf2: " + str(df2.shape))
print("\tdf3: " + str(df3.shape))
print("\tdf4: " + str(df4.shape))
print("\tdf5: " + str(df5.shape))
print("\tdf6: " + str(df6.shape))
print("\tdf7: " + str(df7.shape))
print("\tdfg: " + str(dfg.shape))
print("\tdff: " + str(dff.shape))

# align columns
df1.drop(columns=['frame.time_relative', 'dns.id', 'dns.flags.z', 'dns.flags.rcode', 'dns.qry.class'], inplace=True)
df2.drop(columns=['frame.time_relative', 'dns.id', 'dns.flags.z', 'dns.flags.rcode', 'dns.qry.class'], inplace=True)
df3.drop(columns=['frame.time_relative', 'dns.id', 'dns.flags.z', 'dns.flags.rcode', 'dns.qry.class'], inplace=True)
df4.drop(columns=['frame.time_relative', 'dns.id', 'dns.flags.z', 'dns.flags.rcode', 'dns.qry.class'], inplace=True)
df5.drop(columns=['frame.time_relative', 'dns.id', 'dns.flags.z', 'dns.flags.rcode', 'dns.qry.class'], inplace=True)
df6.drop(columns=['frame.time_relative', 'dns.id', 'dns.flags.z', 'dns.flags.rcode', 'dns.qry.class'], inplace=True)
df7.drop(columns=['frame.time_relative', 'dns.id', 'dns.flags.z', 'dns.flags.rcode', 'dns.qry.class'], inplace=True)
dfg.drop(columns=['dst_AS', 'src_AS'], inplace=True)
dff.drop(columns=['dst_AS', 'src_AS'], inplace=True)
print("Datasets aligned")

# drop unuseful features
df1.drop(columns=['ip.src', 'ip.dst', 'dns.flags.response', 'dns.flags.opcode', 'dns.count.queries',
                  'dns.flags.authenticated', 'dns.flags.truncated'], inplace=True)
df2.drop(columns=['ip.src', 'ip.dst', 'dns.flags.response', 'dns.flags.opcode', 'dns.count.queries',
                  'dns.flags.authenticated', 'dns.flags.truncated'], inplace=True)
df3.drop(columns=['ip.src', 'ip.dst', 'dns.flags.response', 'dns.flags.opcode', 'dns.count.queries',
                  'dns.flags.authenticated', 'dns.flags.truncated'], inplace=True)
df4.drop(columns=['ip.src', 'ip.dst', 'dns.flags.response', 'dns.flags.opcode', 'dns.count.queries',
                  'dns.flags.authenticated', 'dns.flags.truncated'], inplace=True)
df5.drop(columns=['ip.src', 'ip.dst', 'dns.flags.response', 'dns.flags.opcode', 'dns.count.queries',
                  'dns.flags.authenticated', 'dns.flags.truncated'], inplace=True)
df6.drop(columns=['ip.src', 'ip.dst', 'dns.flags.response', 'dns.flags.opcode', 'dns.count.queries',
                  'dns.flags.authenticated', 'dns.flags.truncated'], inplace=True)
df7.drop(columns=['ip.src', 'ip.dst', 'dns.flags.response', 'dns.flags.opcode', 'dns.count.queries',
                  'dns.flags.authenticated', 'dns.flags.truncated'], inplace=True)
dfg.drop(columns=['ip.src', 'ip.dst', 'dns.flags.response', 'dns.flags.opcode', 'dns.count.queries',
                  'dns.flags.authenticated', 'dns.flags.truncated'], inplace=True)
dff.drop(columns=['ip.src', 'ip.dst', 'dns.flags.response', 'dns.flags.opcode', 'dns.count.queries',
                  'dns.flags.authenticated', 'dns.flags.truncated'], inplace=True)
print("Dropped unuseful features columns")

# drop nan values
df1.dropna(axis=0, inplace=True)
df2.dropna(axis=0, inplace=True)
df3.dropna(axis=0, inplace=True)
df4.dropna(axis=0, inplace=True)
df5.dropna(axis=0, inplace=True)
df6.dropna(axis=0, inplace=True)
df7.dropna(axis=0, inplace=True)
dfg.dropna(axis=0, inplace=True)
dff.dropna(axis=0, inplace=True)
print("Samples with nan values dropped")

# transform categorical features to numerical
df1['dns.qry.name'] = df1['dns.qry.name'].apply(lambda a: int(hashlib.sha256(a.encode('utf-8')).hexdigest(), 16) % 10**8)
df2['dns.qry.name'] = df2['dns.qry.name'].apply(lambda a: int(hashlib.sha256(a.encode('utf-8')).hexdigest(), 16) % 10**8)
df3['dns.qry.name'] = df3['dns.qry.name'].apply(lambda a: int(hashlib.sha256(a.encode('utf-8')).hexdigest(), 16) % 10**8)
df4['dns.qry.name'] = df4['dns.qry.name'].apply(lambda a: int(hashlib.sha256(a.encode('utf-8')).hexdigest(), 16) % 10**8)
df5['dns.qry.name'] = df5['dns.qry.name'].apply(lambda a: int(hashlib.sha256(a.encode('utf-8')).hexdigest(), 16) % 10**8)
df6['dns.qry.name'] = df6['dns.qry.name'].apply(lambda a: int(hashlib.sha256(a.encode('utf-8')).hexdigest(), 16) % 10**8)
df7['dns.qry.name'] = df7['dns.qry.name'].apply(lambda a: int(hashlib.sha256(a.encode('utf-8')).hexdigest(), 16) % 10**8)
dfg['dns.qry.name'] = dfg['dns.qry.name'].apply(lambda a: int(hashlib.sha256(a.encode('utf-8')).hexdigest(), 16) % 10**8)
dff['dns.qry.name'] = dff['dns.qry.name'].apply(lambda a: int(hashlib.sha256(a.encode('utf-8')).hexdigest(), 16) % 10**8)

df1['dns.qry.type'] = df1['dns.qry.type'].apply(lambda a: 1 if a == 255 else 0)
df2['dns.qry.type'] = df2['dns.qry.type'].apply(lambda a: 1 if a == 255 else 0)
df3['dns.qry.type'] = df3['dns.qry.type'].apply(lambda a: 1 if a == 255 else 0)
df4['dns.qry.type'] = df4['dns.qry.type'].apply(lambda a: 1 if a == 255 else 0)
df5['dns.qry.type'] = df5['dns.qry.type'].apply(lambda a: 1 if a == 255 else 0)
df6['dns.qry.type'] = df6['dns.qry.type'].apply(lambda a: 1 if a == 255 else 0)
df7['dns.qry.type'] = df7['dns.qry.type'].apply(lambda a: 1 if a == 255 else 0)
dfg['dns.qry.type'] = dfg['dns.qry.type'].apply(lambda a: 1 if a == 255 else 0)
dff['dns.qry.type'] = dff['dns.qry.type'].apply(lambda a: 1 if a == 255 else 0)
print("Transformed categorical features to numerical")

# cast values to int
df1 = df1.astype('int32')
df2 = df2.astype('int32')
df3 = df3.astype('int32')
df4 = df4.astype('int32')
df5 = df5.astype('int32')
df6 = df6.astype('int32')
df7 = df7.astype('int32')
dfg = dfg.astype('int32')
dff = dff.astype('int32')
print("Casted values to int")

# add target column
df1 = df1.assign(target = [1 for i in range(len(df1))])
df2 = df2.assign(target = [1 for i in range(len(df2))])
df3 = df3.assign(target = [1 for i in range(len(df3))])
df4 = df4.assign(target = [1 for i in range(len(df4))])
df5 = df5.assign(target = [1 for i in range(len(df5))])
df6 = df6.assign(target = [1 for i in range(len(df6))])
df7 = df7.assign(target = [1 for i in range(len(df7))])
dfg = dfg.assign(target = [0 for i in range(len(dfg))])
dff = dff.assign(target = [0 for i in range(len(dff))])
print("Added target column")

# feature selection with random forest
concatenated = pd.concat([df1, df2, df3, df4, df5, df6, df7, dfg, dff], ignore_index=True)
concatenated = concatenated.sample(frac=1).reset_index(drop=True)
print("Concatenated dataset shape: " + str(concatenated.shape))

rfc = RandomForestClassifier(class_weight='balanced')
rfc.fit(concatenated.iloc[:,:11], concatenated.iloc[:,11])
importance = rfc.feature_importances_

print("Feature importance:")
for i,v in enumerate(importance):
	print('\tFeature: %0d, Score: %.5f' % (i,v))

names = ['ip.len', 'udp.length', 'dns.flags.authoritative', 'dns.flags.recdesired', 'dns.flags.recavail', 'dns.flags.checkdisable',
         'dns.count.answers', 'dns.count.auth_rr', 'dns.count.add_rr', 'dns.qry.name', 'dns.qry.type']
fig, ax = plt.subplots(figsize=(15,8))
ax.barh(names, importance)
ax.invert_yaxis()
plt.savefig("feature_importance.png")

# drop some more features based on the results of random forest
df1.drop(columns=['dns.flags.checkdisable', 'dns.count.auth_rr'], inplace=True)
df2.drop(columns=['dns.flags.checkdisable', 'dns.count.auth_rr'], inplace=True)
df3.drop(columns=['dns.flags.checkdisable', 'dns.count.auth_rr'], inplace=True)
df4.drop(columns=['dns.flags.checkdisable', 'dns.count.auth_rr'], inplace=True)
df5.drop(columns=['dns.flags.checkdisable', 'dns.count.auth_rr'], inplace=True)
df6.drop(columns=['dns.flags.checkdisable', 'dns.count.auth_rr'], inplace=True)
df7.drop(columns=['dns.flags.checkdisable', 'dns.count.auth_rr'], inplace=True)
dfg.drop(columns=['dns.flags.checkdisable', 'dns.count.auth_rr'], inplace=True)
dff.drop(columns=['dns.flags.checkdisable', 'dns.count.auth_rr'], inplace=True)

print("New datasets shapes:")
print("\tdf1: " + str(df1.shape))
print("\tdf2: " + str(df2.shape))
print("\tdf3: " + str(df3.shape))
print("\tdf4: " + str(df4.shape))
print("\tdf5: " + str(df5.shape))
print("\tdf6: " + str(df6.shape))
print("\tdf7: " + str(df7.shape))
print("\tdfg: " + str(dfg.shape))
print("\tdff: " + str(dff.shape))

# write new data files
df1.to_csv('final_data/booter1.csv', index=False)
df2.to_csv('final_data/booter2.csv', index=False)
df3.to_csv('final_data/booter3.csv', index=False)
df4.to_csv('final_data/booter4.csv', index=False)
df5.to_csv('final_data/booter5.csv', index=False)
df6.to_csv('final_data/booter6.csv', index=False)
df7.to_csv('final_data/booter7.csv', index=False)
dfg.to_csv('final_data/wideg.csv', index=False)
dff.to_csv('final_data/widef.csv', index=False)

print("Done!")
