from sklearn import datasets
import matplotlib.pyplot as plt

from .pca import PCA

data = datasets.load_iris()
X, y = data.data, data.target

pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print("X shape:", X.shape)
print("tranformed shape:", X_projected.shape)

x1, x2 = X_projected[:, 0], X_projected[:, 1]

plt.scatter(
    x1, x2, c=y, edgecolors="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()
