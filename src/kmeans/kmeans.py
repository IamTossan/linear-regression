import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, K=5, max_iters=1000, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize: take random rows as centroids
        self.centroids = X[np.random.choice(X.shape[0], self.K), :]

        # optimize clusters
        for _ in range(self.max_iters):
            self.clusters = self._get_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

        return self._get_cluster_labels(self.clusters)

    def _get_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._get_closest_centroid_index(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    @staticmethod
    def _get_closest_centroid_index(sample, centroids):
        distances = [np.linalg.norm(sample - point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    @staticmethod
    def _is_converged(centroids_old, centroids_current):
        distances = [
            np.linalg.norm(centroids_old[i] - centroids_current[i])
            for i in range(len(centroids_current))
        ]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()
