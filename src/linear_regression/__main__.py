import matplotlib.pyplot as plt

from .linear_regression import LinearRegression

from src.utils import make_random_data, mean_square_error, coef_determination

X, y = make_random_data(n_features=1)
linear_regresion = LinearRegression(learning_rate=0.01)

# Learning phase
linear_regresion.fit(X, y)

# Using model
predictions = linear_regresion.predict(X)
coef = coef_determination(y, predictions)

print("a:", linear_regresion.theta[0, 0])
print("b:", linear_regresion.theta[1, 0])
print("mse: ", mean_square_error(y, predictions))
print("accuracy:", coef)

# Display visuals
plt.scatter(X[:, 0], y)
plt.plot(X[:, 0], predictions, c="r")
plt.show()
