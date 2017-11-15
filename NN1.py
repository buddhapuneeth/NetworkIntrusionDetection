import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
np.random.seed(1337)
train_nsl_kdd_dataset_path = "NSL_KDD_Dataset/KDDTrain+.txt"
col_names = np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"])
attack_dict = {
    'normal': 'normal',
   
    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',
    
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',

    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'worm': 'R2L',
    
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'httptunnel': 'U2R',
    'ps': 'U2R',    
    'sqlattack': 'U2R',
    'xterm': 'U2R'
}
binary_dict = { 'DoS':'attack','Probe':'attack','U2R':'attack','R2L':'attack','normal':'normal'}
categoricalColumns = col_names[[1,2,3]]
binaryColumns = col_names[[6, 11, 13, 14, 20, 21]]
numericColumns = col_names[list(set(range(41)) - set([1,2,3]) - set([6, 11, 13, 14, 20, 21]))]

train = pd.read_csv(train_nsl_kdd_dataset_path, header=None)
train.drop([42], 1, inplace=True)
train.columns = col_names
testData = pd.read_csv("NSL_KDD_Dataset/KDDTest+.txt", header=None)

testData.drop([42], 1, inplace=True)
testData.columns = col_names
mergedDataSet = pd.concat([train, testData]).reset_index(drop = True)
mergedDataSet.shape

# Performing all the encoding
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
enc = LabelEncoder()
#
def encodeCategorical(ser):
    return enc.fit_transform(ser)

mergedDataSet['service'] = encodeCategorical(mergedDataSet['service'])
mergedDataSet['flag'] = encodeCategorical(mergedDataSet['flag'])
mergedDataSet = pd.get_dummies(mergedDataSet, columns=['protocol_type'])
mergedDataSet['labelsMapped'] = mergedDataSet['labels'].map(lambda x: attack_dict[x])


testDataSet = mergedDataSet.loc[train.shape[0]:,:]
trainDataSet = mergedDataSet.loc[:train.shape[0], :]

import sys
from collections import OrderedDict
e = sys.float_info.epsilon
def calAttributeRatio(df, numericColumns,binaryColumns):
    denom = {}
    ar = {}
    for col in numericColumns:
        denom[col] = df[col].mean();

    for col in numericColumns:
        ar[col] = df.fillna(value=0.0).groupby('labelsMapped')[[col]].mean().max().values[0]/(denom[col])

    def test_sum(series):
        return (series.sum()/(len(series)-series.sum()+e))
    for col in binaryColumns:
        groups = df.groupby('labelsMapped')[[col]]
        ar[col] = groups.aggregate([test_sum]).max().values[0]
    return ar

ar_op = calAttributeRatio(trainDataSet,numericColumns,binaryColumns)
def selectTopFeaturesByAR(ar_op, min_ar):
    return [c for c in ar_op.keys() if ar_op[c]>=min_ar]
def normalize(df):
    result = df.copy()
    for col in df.columns:
        max_value = df[col].max()
        min_value = df[col].min()
        result[col] = (df[col] - min_value) / (max_value - min_value+e)
    return result
selectedFeatures = selectTopFeaturesByAR(ar_op,1.00)
train_processed_selectedFeatures = pd.concat([trainDataSet[selectedFeatures], trainDataSet[['labelsMapped', u'protocol_type_icmp', u'protocol_type_tcp', u'protocol_type_udp', u'service', u'flag']]], axis=1)
train_processed_selectedFeatures.head()

test_processed_selectedFeatures = testDataSet[train_processed_selectedFeatures.columns]

print("Data processing done")

x_train, x_val, y_train, y_val = train_test_split(train_processed_selectedFeatures.drop(['labelsMapped'], 1), 
                                                   train_processed_selectedFeatures['labelsMapped'], test_size=0.2, 
                                                   random_state=42)
x_train = normalize(x_train)
print(x_train.shape)
print(y_train.shape)
# print(x_train.applymap(np.isreal))
print("Data partioning")

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
def y_OHE(Y):
    print("Y", Y.shape)
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    
    dummy_y = np_utils.to_categorical(Y)
    return dummy_y


encoder = LabelEncoder()
y_train = y_train.map(lambda x: binary_dict[x])
encoder.fit(y_train)
print("y_train:", y_train.shape)
encoded_Y = encoder.transform(y_train)
print("encoded_Y:", encoded_Y.shape)
dummy_y = np_utils.to_categorical(encoded_Y)
y_val = y_val.map(lambda x: binary_dict[x])
encoder_val = LabelEncoder()
encoder_val.fit(y_val)
encoded_val_Y = encoder_val.transform(y_val)
dummy_y_val = np_utils.to_categorical(encoded_val_Y)

model = Sequential()
model.add(Dense(40, input_dim=38, activation='relu'))
model.add(Dropout(0.4))
# model.add(Dense(20, activation='relu'))
# model.add(Dropout(0.4))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.5, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()
model.get_config()

model.fit(np.array(x_train), np.array(dummy_y), epochs=120, batch_size=200)
# scores = model.evaluate(np.array(x_val), np.array(dummy_y_val))
# print(scores)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scores = model.evaluate(np.array(x_train), np.array(dummy_y))
print(scores)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
from sklearn.metrics import accuracy_score,precision_score, recall_score
print("#####VALIDATION SCORES#####")
def estimations(y_val,pred):
    print('Accuracy Score', accuracy_score(y_val, pred))
    print('Precision Score', precision_score(y_val, pred, average = "weighted"))
    print('Recall Score', recall_score(y_val, pred, average = "weighted"))
    print(pd.crosstab(y_val,pred, rownames=['True'], colnames=['Predicted'], margins=True))

pred = model.predict(np.array(x_val),batch_size=200)
pred_ds = pd.DataFrame(pred)
dummy_y_val_ds = pd.DataFrame(dummy_y_val)
estimations(np.array(pred_ds.idxmax(1)),np.array(dummy_y_val_ds.idxmax(1)))
print(np.array(pred_ds.shape))
print(np.array(pred_ds.idxmax(1)).shape)
print(np.array(dummy_y_val_ds.shape))
print(np.array(dummy_y_val_ds.idxmax(1)).shape)
print("#####TEST SCORES#####")
x_test = test_processed_selectedFeatures.drop(['labelsMapped'], 1)
y_test = test_processed_selectedFeatures['labelsMapped']
print("x_test:", x_test.shape)
print("y_test:", y_test.shape)
# print("y_test_values:", y_test)
y_test = y_test.map(lambda x: binary_dict[x])
encoder_test = LabelEncoder()
encoder_test.fit(y_test)
encoded_test_Y = encoder_test.transform(y_test)
dummy_y_test = np_utils.to_categorical(encoded_test_Y)
print("encoded_test_Y:", encoded_test_Y.shape)
# print("encoded_test_Y_values:",encoded_test_Y)
print("dummy_y_test:",dummy_y_test.shape)
pred_test = model.predict(np.array(x_test),batch_size=200)
print("pred_test:", pred_test.shape)
pred_test_ds = pd.DataFrame(pred_test)
dummy_y_test_ds = pd.DataFrame(dummy_y_test)
#print(pred_test_ds)
print(np.array(pred_test_ds.shape))
#print(dummy_y_test_ds)
print(np.array(pred_test_ds.idxmax(1)).shape)
print(np.array(dummy_y_test_ds.shape))
print(np.array(dummy_y_test_ds.idxmax(1)).shape)
estimations(np.array(pred_test_ds.idxmax(1)),np.array(dummy_y_test_ds.idxmax(1)))