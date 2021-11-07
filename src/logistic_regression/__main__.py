from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

from .logistic_regression import LogisticRegression

from src.utils import coef_determination

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

logistic_regression = LogisticRegression(learning_rate=0.0001, n_iters=1000)
logistic_regression.fit(X_train, y_train.reshape((y_train.shape[0], 1)))

predictions = logistic_regression.predict(X_test)


def accuracy(predicted_labels, actual_labels):
    return np.sum(predicted_labels == actual_labels) / len(actual_labels)


print("LR classification accuracy:", accuracy(y_test, predictions))
