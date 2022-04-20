'''
hw4 - problem 2

References:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    textbook - introduction to machine learning with python, a guide for data scientists bu andreas c. muller and sarah guido
'''

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

#import data
train_input = pd.read_csv(r'C:/Users/Berto/data for homework 3/problem 4/Data for Problem 4_train_input.csv')
train_labels = pd.read_csv(r'C:/Users/Berto/data for homework 3/problem 4/Data for Problem 4_train_labels.csv')
test_input = pd.read_csv(r'C:/Users/Berto/data for homework 3/problem 4/Data for Problem 4_test_input.csv')

#combine data
frames_train = [train_input, train_labels]
df_train = pd.concat(frames_train, axis=1, join='inner')

#seperate data
X = df_train[['x1', 'x2', 'x3','x4']]
y = df_train['y1']

#convert to array
X_train, y_train = X.to_numpy(), y.to_numpy()
X_test = test_input.to_numpy()

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)

mlp.fit(X_train, y_train)

MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(13, 13, 13), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)

y_test_predictions = mlp.predict(X_test)

#hw4 - problem 2 starts

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Accuracy on training set: {:.3}".format(forest.score(X_train, y_train)))
print("Accuracy on testing set: {:.3}".format(forest.score(X_test, y_test_predictions)))

