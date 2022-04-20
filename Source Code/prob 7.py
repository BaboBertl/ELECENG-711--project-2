'''
hw4 - problem 6

References:
    textbook - introduction to machine learning with python, a guide for data scientists bu andreas c. muller and sarah guido
    https://www.geeksforgeeks.org/linear-regression-implementation-from-scratch-using-python/
'''

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
  
# Linear Regression
class LinearRegression():
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
          
    # Function for model training
    def fit(self, X, Y):
 
        # no_of_training_examples, no_of_features
        self.m, self.n = X.shape
       
        # weight initialization
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
         
        # gradient descent learning
        for i in range(self.iterations):
            self.update_weights()
        return self
      
    # Helper function to update weights in gradient descent
    def update_weights(self):
        Y_pred = self.predict(self.X )
          
        # calculate gradients  
        dW = - (2 * (self.X.T).dot( self.Y - Y_pred)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m 
          
        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self
      
    # Hypothetical function  h(x) 
    def predict(self, X) :
        return X.dot(self.W) + self.b

# driver code
def main() :
        
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

    X_test = df_test[['x1',]]
    y_test = df_test['y1']

    #convert dataframe to array
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    X_test, y_test = X_test.to_numpy(), y_test.to_numpy()
    
    # Model training
    model = LinearRegression(iterations = 1000, learning_rate = 0.01)
    model.fit(X_train, y_train)
      
    # Prediction on test set
    Y_pred = model.predict( X_test )
    print("Predicted values ", np.round(Y_pred[:3], 2)) 
    print("Real values      ", y_test[:3])
    print("Trained W        ", round(model.W[0], 2))
    print("Trained b        ", round(model.b, 2))
      
    # Visualization on test set 
    plt.scatter(X_test, y_test, color = 'blue')
    plt.scatter(X_train, y_train, color = 'red')
    plt.plot(X_test, Y_pred, color = 'orange')
    plt.legend(["linear regression", "testing data", "training data"])
    plt.show()
     
if __name__ == "__main__" : 
    main()
    
    