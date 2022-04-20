'''
Problem 8
'''

import mglearn
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

#import data
x_train = pd.read_csv(r'C:/Users/Berto/hw4/Data/Data for Problem 7/prob 7 - trainx.csv')
y_train = pd.read_csv(r'C:/Users/Berto/hw4/Data/Data for Problem 7/prob 7 - trainy.csv')

x_test = pd.read_csv(r'C:/Users/Berto/hw4/Data/Data for Problem 7/prob 7 - testx.csv')
y_test = pd.read_csv(r'C:/Users/Berto/hw4/Data/Data for Problem 7/prob 7 - testy.csv')

#combine data
frames_train = [x_train, y_train]
df_train = pd.concat(frames_train, axis=1, join='inner')

frames_test = [x_test, y_test]
df_test = pd.concat(frames_test, axis=1, join='inner')

#separate data
X_train = df_train[['x1']]
y_train = df_train['y1']

X_test = df_test[['x1']]
y_test = df_test['y1']

#convert dataframe to array
X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

#creating neural network
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp1 = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)

mglearn.discrete_scatter(X_train, y_train)

