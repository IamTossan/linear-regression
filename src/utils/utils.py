import numpy as np
from sklearn.datasets import make_regression


def make_random_data(n_features=1):
    x, y = make_regression(n_samples=100, n_features=n_features, noise=10)
    y = y.reshape(y.shape[0], 1)

    return x, y


def mean_square_error(y_actual, y_pred):
    return np.mean((y_actual - y_pred) ** 2)


def cost_function(prediction, actual):
    """returns a number"""
    m = len(actual)
    return 1 / (2 * m) * np.sum((prediction - actual) ** 2)


# R^2
def coef_determination(y, pred):
    u = ((y - pred) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    return 1 - u / v
