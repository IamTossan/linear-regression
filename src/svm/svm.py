import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        print(y_)

        self.w = np.zeros(n_features)
        self.b = 0

        for i in range(self.n_iters):

            for idx, x_i in enumerate(X):

                isCorrectlyClassified = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                dw = (2 * self.lambda_param * self.w) - (
                    0 if isCorrectlyClassified else np.dot(x_i, y_[idx])
                )
                db = 0 if isCorrectlyClassified else y_[idx]

                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
