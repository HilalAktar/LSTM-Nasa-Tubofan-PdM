
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import tensorflow as tf

from tensorflow import keras
from keras.layers import Dense, LSTM, Input, Dropout, TimeDistributed, RepeatVector, Activation
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from PdM_LSTM_Functions import gen_label,gen_sequence,prob_failure

###############data preparations
columns1 = ["test_results", "null"]
columns = ["id", "cycle", "op1", "op2", "op3", "sensor1", "sensor2", "sensor3", "sensor4",
           "sensor5", "sensor6", "sensor7", "sensor8", "sensor9", "sensor10", "sensor11",
           "sensor12", "sensor13", "sensor14", "sensor15", "sensor16", "sensor17", "sensor18", "sensor19",
           "sensor20", "sensor21", "sensor22", "sensor23"]

# Dosya yollarını güncelle
train_path = r'train_FD001.txt'
test_path = r'test_FD001.txt'
test_results_path = r'RUL_FD001.txt'

# Verileri okutrain = pd.read_csv(train_path, sep=" ", names=columns)
train= pd.read_csv(train_path, sep=" ", names=columns)
test = pd.read_csv(test_path, sep=" ", names=columns)
test_results = pd.read_csv(test_results_path, sep=" ", names=columns1)

train.info()
test.info()
test_results.info()

print(f"Number of unique engine ids in train set: {len(train.id.unique())}")
print(f"Number of unique engine ids in test set: {len(test.id.unique())}")

train = train.drop(["sensor22", "sensor23"], axis=1)
test = test.drop(["sensor22", "sensor23"], axis=1)

test.isnull().values.any()
train.isnull().values.any()
test_results.isnull().values.any()

test_results


###############explantory analysis

test_results = test_results.drop(["null"], axis=1)
test_results

colnum=len(train.columns)
colnum


train=train.astype(float)

train.columns


import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

train.value_counts()
#sns.pairplot(test,hue='cycle')
train.plot(kind='line', x='id', y='cycle',alpha =1,color = 'pink')
plt.xlabel('id')
plt.ylabel('Cycle time')
plt.title('Number of Cycles of Different Engines')

###############Multicollinearity, normal dist. and corrolation

#sns.pairplot(test.drop("id", axis=1), hue="cycle", size=3)

subset_stats = train.agg(['mean', 'std']).T[2:]
ax = subset_stats.plot.bar(figsize=(18,10),
                           y="std",color = 'pink')
ax.set_title("Standard Deviation of Every Sensor From Every Engine", fontsize=22)
ax.set_xlabel("Sensor", fontsize=18)
ax.set_ylabel("Standard Deviation", fontsize=18)

for p in ax.patches:
    ax.annotate(str(round(p.get_height(),3)), (p.get_x() * 1.005, p.get_height() * 1.005))

op_set=["op"+str(i) for i in range(1,4)]
sensor=["sensor"+str(i) for i in range(1,22)]

sns.pairplot(train, hue="cycle",x_vars=op_set,y_vars=sensor)

train.drop(["op3","sensor1","sensor5","sensor6","sensor10","sensor16","sensor18","sensor19"],axis=1,inplace=True)

corr = train.drop(["id", "cycle", "op1", "op2"], axis=1).corr()
fig, ax = plt.subplots()
fig.set_size_inches(12,11)
sns.heatmap(corr, annot=True, cmap="RdPu", center=0, ax=ax)

corr = train.drop(["sensor4","sensor7","id", "cycle", "op1", "op2","sensor11","sensor9","sensor12","sensor8","sensor15"], axis=1).corr()
fig, ax = plt.subplots()
fig.set_size_inches(12,11)
sns.heatmap(corr, annot=True, cmap="RdPu", center=0, ax=ax)


train.drop(["sensor4","sensor7","sensor11","sensor9","sensor12","sensor8","sensor15"],axis=1,inplace=True)
test.drop(["sensor4","sensor7", "op3","sensor1","sensor5","sensor6","sensor10","sensor16","sensor18","sensor19",
           "sensor11","sensor9","sensor12","sensor8","sensor15"],axis=1,inplace=True)

print(sns.boxplot(x = train['sensor2']))
print(train["sensor2"])
print(sns.boxplot(x = train['sensor3']))
print(sns.boxplot(x = train['sensor13']))
print(sns.boxplot(x = train['sensor14']))
print(sns.boxplot(x = train['sensor17']))
print(sns.boxplot(x = train['sensor20']))
print(sns.boxplot(x = train['sensor21']))

###############Sensor datası için outlier analizi
n = 0
for n in [2, 3, 13, 14, 17, 20, 21]:
    train_sensor = train["sensor" + str(n)]

    Q3, Q1 = np.percentile(train_sensor, [75, 25])
    IQR = Q3 - Q1
    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR

    (train_sensor > ust_sinir) | (train_sensor < alt_sinir)
    aykiri_ust = (train_sensor > ust_sinir)
    aykiri_alt = (train_sensor < alt_sinir)
    train_sensor[aykiri_ust] = ust_sinir
    train_sensor[aykiri_alt] = alt_sinir
    n = n + 1

print(sns.boxplot(x = train['sensor2']))
print(sns.boxplot(x = train['sensor3']))
print(sns.boxplot(x = train['sensor13']))
print(sns.boxplot(x = train['sensor14']))
print(sns.boxplot(x = train['sensor17']))
print(sns.boxplot(x = train['sensor20']))
print(sns.boxplot(x = train['sensor21']))

######################LSTM############################
test_results=pd.read_csv(r'C:\Users\isilh\OneDrive\Masaüstü\tez dökümanları\NASA pdm\RUL_FD001.txt',sep=" ",header=None).drop([1],axis=1)
test_results.columns=['more']
test_results['id']=test_results.index+1
test_results.head()

rul = pd.DataFrame(test.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
rul.head()

train.head()

# run to failure
test_results['rtf']=test_results['more']+rul['max']
test_results.head()

test_results.drop('more', axis=1, inplace=True)
test = test.merge(test_results, on=['id'], how='left')
test['ttf']= test['rtf'] - test['cycle']
test.drop('rtf', axis=1, inplace=True)
test.head()

train['ttf'] = train.groupby(['id'])['cycle'].transform(max)-train['cycle']
train.head()

train=train.copy()
test=test.copy()
period=30
train['label_bc'] = train['ttf'].apply(lambda x: 1 if x <= period else 0)
test['label_bc'] = test['ttf'].apply(lambda x: 1 if x <= period else 0)
train.head()

features_col_name=["id","cycle","op1","op2","sensor2","sensor3","sensor13","sensor14","sensor17"
         ,"sensor20","sensor21"]
target_col_name='label_bc'

seq_length=10
seq_cols=features_col_name

X_train=np.concatenate(list(list(gen_sequence(train[train['id']==id], seq_length, seq_cols)) for id in train['id'].unique()))
print(X_train.shape)

y_train=np.concatenate(list(list(gen_label(train[train['id']==id], 50, seq_cols,'label_bc')) for id in train['id'].unique()))
print(y_train.shape)

X_test=np.concatenate(list(list(gen_sequence(test[test['id']==id], seq_length, seq_cols)) for id in test['id'].unique()))
print(X_test.shape)

y_test=np.concatenate(list(list(gen_label(test[test['id']==id], 50, seq_cols,'label_bc')) for id in test['id'].unique()))
print(y_test.shape)

nb_features =X_train.shape[2]
timestamp=seq_length

model = Sequential()

model.add(LSTM(
         input_shape=(timestamp, nb_features),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
          units=60,
          return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# fit the network
model.fit(X_train, y_train, epochs=10, batch_size=400, validation_split=0.05, verbose=1,
          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0)])

scores = model.evaluate(X_train, y_train, verbose=1, batch_size=400)
print('Accurracy: {}'.format(scores[1]))


y_pred = model.predict(X_test)
y_pred = np.round(y_pred).astype(int)
print('Accuracy of model on test data: ',accuracy_score(y_test,y_pred))

machine_id=16
print('Probability that machine will fail within 30 days: ',prob_failure(machine_id))