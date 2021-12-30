# -*- coding: utf-8 -*-

# -- Sheet --

#importing essential libraries that is required in this exercise.
import numpy as np
import matplotlib.pyplot as plt

# # Creating random data-set
# Following the tutorial, I am using np.random to generate random number data set for this exercise.


np.random.seed() #setting seed(0). Initialising seed here makes predictable random numbers
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# Plotting this data


plt.scatter(x, y, s=10) #s = 10, the size paramete
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Let's try now to implement Linear regression with the help of Scikit-learn library. The tutorial guy kind of skipped the explanation of this part, so IDK how exactly I am going to approach this. However, I will include clarification whenever I get it for respective parts


#importing required modules from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#MODEL INITALIZATION
regression_model = LinearRegression()

#FITTING THE DATA or TRAINING THE MODEL
regression_model.fit(x,y)

#PREDICT
y_predicted = regression_model.predict(x)


# So there is actually a part here where the tutorial guy is calculating rmse(root mean square error I think)and r2 score. For now, lets skip this part and try to plot what we got.


plt.scatter(x,y,s=10)
plt.xlabel('x')
plt.ylabel('y')

plt.plot(x,y_predicted,color='r')
plt.show()

# -- Direct Linear Regression implementation --

#DIDNT WORK LAST TIME I TRIED
# imports
import numpy as np


class LinearRegressionUsingGD:
    """Linear Regression Using Gradient Descent.
    Parameters
    ----------
    eta : float
        Learning rate
    n_iterations : int
        No of passes over the training set
    Attributes
    ----------
    w_ : weights/ after fitting the model
    cost_ : total error of the model after each iteration
    """

    def __init__(self, eta=0.05, n_iterations=1000):
        self.eta = eta
        self.n_iterations = n_iterations

    def fit(self,x,y):
        """Fit the training data
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Training samples
        y : array-like, shape = [n_samples, n_target_values]
            Target values
        Returns
        -------
        self : object
        """
    

        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        m = x.shape[0]

        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.eta / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)
        return self

    def predict(self, x):
        """ Predicts the value after the model has been trained.
        Parameters
        ----------
        x : array-like, shape = [n_samples, n_features]
            Test samples
        Returns
        -------
        Predicted value
        """
        return np.dot(x, self.w_)
np.random.seed(0) #setting seed(0). Initialising seed here makes predictable random numbers
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)
#print(x)
#print('y is')
#print(y)
f = LinearRegressionUsingGD(2,50)
f.fit(x,y)

f.predict(0.68270507)
 

