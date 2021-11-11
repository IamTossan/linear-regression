import numpy as np
from sklearn import datasets

from .kmeans import KMeans

X, y = datasets.make_blobs(
    n_samples=500, n_features=2, centers=5, random_state=40, shuffle=True
)

clusters = len(np.unique(y))
print("cluster amount:", clusters)

k = KMeans(K=clusters, max_iters=150, plot_steps=True)
y_pred = k.predict(X)

k.plot()
