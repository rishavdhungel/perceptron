import numpy as np 

def unit_step_func(x):
        return np.where(x > 0, 1,0)

class Perceptron:
    def __init__(self,learning_rate = 0.01, iterations = 1000):

        self.learning_rate = learning_rate
        self.iterations = iterations
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        no_of_samples, no_of_features = X.shape
        self.weights = np.zeros(no_of_features)
        self.bias = 0
        
        #class lables
        y_ = np.where(y > 0, 1,0)
        #optimization
        for _ in range(self.iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                #perception update
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X,self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted