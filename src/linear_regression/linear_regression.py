import numpy as np


class LinearRegression:
    @staticmethod
    def addOnesColumn(X):
        return np.hstack((X, np.ones((X.shape[0], 1))))

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.theta = None

    def model(self, X):
        return X.dot(self.theta)

    def fit(self, rawX, y):
        n_samples, n_features = rawX.shape

        self.theta = np.zeros((n_features + 1, 1))
        X = self.addOnesColumn(rawX)

        for _ in range(self.n_iters):
            y_predicted = self.model(X)
            self.theta -= self.lr * (1 / n_samples) * X.T.dot(y_predicted - y)

    def predict(self, X):
        y_approximated = self.model(self.addOnesColumn(X))
        return y_approximated
