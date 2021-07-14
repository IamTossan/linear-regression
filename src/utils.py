import numpy as np
from sklearn.datasets import make_regression


def make_random_data():

    x, y = make_regression(n_samples=100, n_features=2, noise=10)
    y = y.reshape(y.shape[0], 1)

    xs = np.hstack((x, np.ones((x.shape[0], 1))))

    theta = np.random.randn(3, 1)

    return xs, y, theta
