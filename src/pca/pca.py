import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # variance/covariance matrix
        cov = np.cov(X.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov)

        idxs = np.argsort(eigenvalues)[::-1]

        # get n eigenvectors sorted by eigenvalues DESC
        self.components = eigenvectors.T[idxs][0 : self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)
