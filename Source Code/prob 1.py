'''
hw4 - problem 1

References:
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    textbook - introduction to machine learning with python, a guide for data scientists bu andreas c. muller and sarah guido
'''

import matplotlib.pyplot as plt
import mglearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

#import data
x_train = pd.read_csv(r'C:/Users/Berto/data for homework 3/problem 7/Problem 7_data_trainx.csv')
y_train = pd.read_csv(r'C:/Users/Berto/data for homework 3/problem 7/Problem 7_data_trainy.csv')

x_test = pd.read_csv(r'C:/Users/Berto/data for homework 3/problem 7/Problem 7_data_testx.csv')
y_test = pd.read_csv(r'C:/Users/Berto/data for homework 3/problem 7/Problem 7_data_testy.csv')

#combine data
frames_train = [x_train, y_train]
df_train = pd.concat(frames_train, axis=1, join='inner')

frames_test = [x_test, y_test]
df_test = pd.concat(frames_test, axis=1, join='inner')

#separate data
X_train = df_train[['x1', 'x2']]
y_train = df_train['y1']

X_test = df_test[['x1', 'x2']]
y_test = df_test['y1']

#convert dataframe to array
X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

#generate forest
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(),forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
    
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)

axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

print("Accuracy on training set: {:.3}".format(forest.score(X_train, y_train)))
print("Accuracy on testing set: {:.3}".format(forest.score(X_test, y_test)))

