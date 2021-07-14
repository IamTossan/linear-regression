import numpy as np


def model(X, theta):
    """returns a m * 1 sized matrix of predictions"""
    return X.dot(theta)


def cost_function(prediction, actual):
    """returns a number"""
    m = len(actual)
    return 1 / (2 * m) * np.sum((prediction - actual) ** 2)


def grad(X, pred, y):
    """returns a (n+1) * 1 matrix"""
    m = len(y)
    return 1 / m * X.T.dot(pred - y)


def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        pred = model(X, theta)
        theta = theta - learning_rate * grad(X, pred, y)
        cost_history[i] = cost_function(pred, y)
    return theta, cost_history


# R^2
def coef_determination(y, pred):
    u = ((y - pred) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    return 1 - u / v
