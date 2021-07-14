import matplotlib.pyplot as plt

from linear_regression import model, coef_determination, gradient_descent
from utils import make_random_data


xs, y, theta = make_random_data()

# Learning phase
theta_final, cost_history = gradient_descent(
    xs, y, theta, learning_rate=0.01, n_iterations=1000
)

# Using model
predictions = model(xs, theta_final)
coef = coef_determination(y, predictions)

print("theta final: ", theta_final)
print("coef: ", coef)

# Display visuals
plt.scatter(xs[:, 0], y)
plt.plot(xs[:, 0], predictions, c="r")
plt.show()

plt.plot(range(1000), cost_history)
plt.show()
