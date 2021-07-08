import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# Dataset

x, y = make_regression(n_samples=100, n_features=2, noise=10)

print(x.shape)
y = y.reshape(y.shape[0], 1)
print(y.shape)

# make data more polynomial
# y = y + abs(y/2)

# matrix X

X = np.hstack((x, np.ones((x.shape[0], 1))))

# X = np.hstack((x**2, X))

theta = np.random.randn(3, 1)
print("theta initial: ", theta)

# Modele


def model(X, theta):
    return X.dot(theta)


fig, ax = plt.subplots()
ax.scatter(x[:, 0], y)
ax.plot(x, model(X, theta), c="r")
plt.show()

# fonction cout
def cost_function(X, y, theta):
    m = len(y)
    return 1 / (2 * m) * np.sum((model(X, theta) - y) ** 2)


# gradient descent


def grad(X, y, theta):
    m = len(y)
    return 1 / m * X.T.dot(model(X, theta) - y)


def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        theta = theta - learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)
    return theta, cost_history


# R^2
def coef_determination(y, pred):
    u = ((y - pred) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    return 1 - u / v


# Main

theta_final, cost_history = gradient_descent(
    X, y, theta, learning_rate=0.01, n_iterations=1000
)

predictions = model(X, theta_final)
coef = coef_determination(y, predictions)
print("theta final: ", theta_final)
print("coef: ", coef)

plt.scatter(x[:, 0], y)
plt.plot(x[:, 0], predictions, c="r")
plt.show()

plt.plot(range(1000), cost_history)
plt.show()
