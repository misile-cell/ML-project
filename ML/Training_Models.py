# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:13:37 2023

@author: Misile
"""

# Training Models



# Linear Regression model



# In this chapter, we will start by looking at the Linear Regression model,
# one of the simplest models there is. We will discuss two very different
# ways to train it:

# (i) Using a direct “closed-form” equation that directly computes the model 
# parameters that best fit the model to the training set (i.e., the model 
# parameters that minimize the cost function over the training set).

# (ii) Using an iterative optimization approach, called Gradient Descent (GD),
# that gradually tweaks the model parameters to minimize the cost function 
# over the training set

# The model: Y = Theta*X, and the value of theta_hat that minimizes this equation
# is called the normal equation: Theta_hat = (X^TX)^{-1}X*Y. Let’s generate some
# linear-looking data to test this equation

import numpy as np
import matplotlib.pyplot as plt 

X = 2 * np.random.rand(100, 1) 
y = 4 + 3 * X + np.random.randn(100, 1)

# Now let’s compute θ using the Normal Equation

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance 
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) 

# theta_best = (4.20, 3.04)

# The LinearRegression class is based on the scipy.linalg.lstsq() 
# function (the name stands for “least squares”), which you could 
# call directly

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6) 
theta_best_svd 

# This function computes θ = X^+y, where X+ is the pseudoinverse of X^+ 
# (specifically the Moore-Penrose inverse)

np.linalg.pinv(X_b).dot(y)

# The pseudoinverse itself is computed using a standard matrix factorization 
# technique called Singular Value Decomposition (SVD) that can decompose the 
# training set matrix X into the matrix multiplication of three matrices U Σ V^T

# This approach is more efficient than computing the Normal Equation, plus it
# handles edge cases nicely: indeed, the Normal Equation may not work if the 
# matrix X^TX is not invertible (i.e., singular), such as if m < n or 
# if some features are redundant, but the pseudoinverse is always defined. 

# Computational Complexity: the Normal Equation method is not suited for large 
# data sets than the SVD method.

#  different ways to train a Linear Regression model, better suited for cases 
# where there are a large number of features, or too many training instances 
# to fit in memory. 

# 1. Gradient Descent: is a very generic optimization algorithm capable of
# finding optimal solutions to a wide range of problems. The general idea of 
# Gradient Descent is to tweak parameters iteratively in order to minimize a 
# cost function.

# Gradient Descent does: it measures the local gradient of the error function
# with regards to the parameter vector 
# An important parameter in Gradient Descent is the size of the steps, 
# determined by the learning rate hyperparameter

# Pitfalls: high learning rate and not all cost functions are nice and regular
# functions

# 2. Batch Gradient Descent: θ^{next step} = θ−η∇_θMSE(θ), ie, calculate the 
# partial derivatives at once, multiply by the learning rate, subtract that from the 
# vector of parameters. Multiplying the gradient vector by η determines the size 
# of the downhill step 

# Let’s look at a quick implementation of this algorithm

eta = 0.1  # learning rate 
n_iterations = 1000 
m = 100
theta = np.random.randn(2,1)  # random initialization
for iteration in range(n_iterations):    
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)    
    theta = theta - eta * gradients

# To find a good learning rate, you can use grid search (see Chapter 2). 
# However, you may want to limit the number of iterations so that grid search 
# can eliminate models that take too long to converge.
    
#  A simple solution is to set a very large number of iterations but to 
# interrupt the algorithm when the gradient vector becomes tiny; tolerance level

# 3. Stochastic Gradient Descent: picks a random instance in the training set at 
# every step and computes the gradients based only on that single instance. 

# On the other hand, due to its stochastic (i.e., random) nature, this algorithm 
# is much less regular than Batch Gradient Descent: instead of gently decreasing
# until it reaches the minimum, the cost function will bounce up and down, 
# decreasing only on average

# This code implements Stochastic Gradient Descent using a simple learning 
# schedule


n_epochs = 50 
t0, t1 = 5, 50  # learning schedule hyperparameters

def learning_schedule(t):    
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):    
    for i in range(m):        
        random_index = np.random.randint(m)        
        xi = X_b[random_index:random_index+1]        
        yi = y[random_index:random_index+1]        
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)        
        eta = learning_schedule(epoch * m + i)        
        theta = theta - eta * gradients

# To perform Linear Regression using SGD with Scikit-Learn, you can use the 
# SGDRegressor class, which defaults to optimizing the squared error cost 
# function
 # max_iter is _epoch       
        
from sklearn.linear_model import SGDRegressor
 
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1) 
sgd_reg.fit(X, y.ravel())

sgd_reg.intercept_, sgd_reg.coef_

# 4. Mini-batch Gradient Descent:  at each step, instead of computing the 
# gradients based on the full training set (as in Batch GD) or based on just 
# one instance (as in Stochastic GD), Minibatch GD computes the gradients on 
# small random sets of instances called minibatches. The main advantage of 
# Mini-batch GD over Stochastic GD is that you can get a performance boost 
# from hardware optimization of matrix operations, especially when using GPUs. 

#  Mini-batch GD will end up walking around a bit closer to the minimum than SGD




# Polynomial Regression


# What if your data is actually more complex than a simple straight line? 
# Surprisingly, you can actually use a linear model to fit nonlinear data. 
# A simple way to do this is to add powers of each feature as new features, 
# then train a linear model on this extended set of features. This technique 
# is called Polynomial Regression.


m = 100 
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

from sklearn.preprocessing import PolynomialFeatures 

poly_features = PolynomialFeatures(degree=2, include_bias=False) 
X_poly = poly_features.fit_transform(X)

X[0], X_poly[0]

# X_poly now contains the original feature of X plus the square of this feature

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression() 
lin_reg.fit(X_poly, y) 
lin_reg.intercept_, lin_reg.coef_

#  How can you tell that your model is overfitting or underfitting the data? 

# Learning Curves 

#  The following code defines a function that plots the learning curves of a 
# model given some training data:

from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)    
    train_errors, val_errors = [], []    
    for m in range(1, len(X_train)):        
        model.fit(X_train[:m], y_train[:m])        
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)        
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))        
        val_errors.append(mean_squared_error(y_val, y_val_predict))    
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=1, label="train")    
    plt.plot(np.sqrt(val_errors), "b-", linewidth=1, label="val") 


# How to interpret the curves
    
plot_learning_curves(lin_reg, X, y)
    
# Underfitting:
 
# training data
    
# when there are just one or two instances in the training set, the model can 
# fit them perfectly, which is why the curve starts at zero. But as new 
# instances are added to the training set, it becomes impossible for the model
# to fit the training data perfectly, both because the data is noisy and 
# because it is not linear at all. 
# the error on the training data goes up until it reaches a 
# plateau, at which point adding new instances to the training set doesn’t 
# make the average error much better or worse. 

# Validating data
    
# When the model is trained on very few training instances, it is incapable of
# generalizing properly, which is why the validation error is initially quite
# big. Then as the model is shown more training examples, it learns and thus
# the validation error slowly goes down. However, once again a straight line
# cannot do a good job modeling the data, so the error ends up at a plateau,
# very close to the other curve.

# Overfitting:

from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([        
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),  
        ("lin_reg", LinearRegression()),    
        ])
    
plot_learning_curves(polynomial_regression, X, y)

# There is a gap between the curves. This means that the model performs 
# significantly better on the training data than on the validation data, 
# which is the hallmark of an overfitting model. However, if you used a 
# much larger training set, the two curves would continue to get closer.



# Regularized Linear Models


# The solution to overfitting is the regularization of the model.  
# A simple way to regularize a polynomial model is to reduce the number 
# of polynomial degrees. 

# 1. Ridge Regression (regularized Linear Regression) 

# Closed form fuction is θ = (XTX+αA)^−1 X^T y

# Ridge Regression with Scikit-Learn using a closed-form solution

from sklearn.linear_model import Ridge 
ridge_reg = Ridge(alpha=1, solver="cholesky") 
ridge_reg.fit(X, y)

ridge_reg.predict([[1.5]])

# And using Stochastic Gradient Descent

sgd_reg = SGDRegressor(penalty="l2") 
sgd_reg.fit(X, y.ravel()) 
sgd_reg.predict([[1.5]])

# The penalty hyperparameter sets the type of regularization term to use. 
# Specifying "l2" indicates that you want SGD to add a regularization term to 
# the cost function equal to half the square of the ℓ2 norm of the weight 
# vector: this is simply Ridge Regression


# 2. Lasso Regression (regularized Linear Regression)

# Least Absolute Shrinkage and Selection Operator Regression (simply called 
# Lasso Regression) is another regularized version of Linear Regression: just 
# like Ridge Regression, it adds a regularization term to the cost function, 
# but it uses the ℓ1 norm of the weight vector instead of half the square of 
# the ℓ2 norm

# Here is a small Scikit-Learn example using the Lasso class. Note that you 
# could instead use an SGDRegressor(penalty="l1").

from sklearn.linear_model import Lasso 
lasso_reg = Lasso(alpha=0.1) 
lasso_reg.fit(X, y) 
lasso_reg.predict([[1.5]]) 

# 3. Elastic Net 

# Elastic Net is a middle ground between Ridge Regression and Lasso Regression.
# The regularization term is a simple mix of both Ridge and Lasso’s regularization 
# terms, and you can control the mix ratio r. When r = 0, Elastic Net is equivalent 
# to Ridge Regression, and when r = 1, it is equivalent to Lasso Regression

# Here is a short example using Scikit-Learn’s ElasticNet (l1_ratio corresponds
# to the mix ratio r):

from sklearn.linear_model import ElasticNet 
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5) 
elastic_net.fit(X, y) 
elastic_net.predict([[1.5]]) 

# 4. Early Stopping 

# A very different way to regularize iterative learning algorithms such as 
# Gradient Descent is to stop training as soon as the validation error reaches 
# a minimum. This is called early stopping.

# Here is a basic implementation of early stopping:

from sklearn.base import clone 
from sklearn.preprocessing import StandardScaler


# prepare the data 
poly_scaler = Pipeline([        
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),        
        ("std_scaler", StandardScaler())    
        ]) 

####################################
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)    
#########################################   
    
X_train_poly_scaled = poly_scaler.fit_transform(X_train) 
X_val_poly_scaled = poly_scaler.transform(X_val)
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,                       
                       penalty=None, learning_rate="constant", eta0=0.0005)

minimum_val_error = float("inf") 
best_epoch = None 
best_model = None 
for epoch in range(1000):    
    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off    
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)    
    val_error = mean_squared_error(y_val, y_val_predict)    
    if val_error < minimum_val_error:        
        minimum_val_error = val_error        
        best_epoch = epoch        
        best_model = clone(sgd_reg)











