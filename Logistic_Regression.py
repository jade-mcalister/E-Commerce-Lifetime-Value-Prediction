import numpy as np
from Main import X_train, X_test, y_train, y_test


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:

    # constructor
    def __init__(self, learning_rate=0.05):
        self.learning_rate = learning_rate
        # initializing weights and bias
        self.weights = None
        self.bias = 0

    # sigmoid function

    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.weights is None:
            self.weights = np.zeros(n_features, dtype=np.float128)

        # gradient descent
        linear_combination = np.dot(X, self.weights) + self.bias  # y=wx+b
        y_predicted = sigmoid(linear_combination)  # implements sigmoid function

        # gradient derivatives for weights and bias
        dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
        db = (1 / n_samples) * np.sum(y_predicted - y)

        # update weights
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_combination = np.dot(X, self.weights) + self.bias  # y=wx+b
        y_predicted = sigmoid(linear_combination)

        # rounds to zero or 1
        y_predicted_label = []
        for n in range(len(y_predicted)):
            if y_predicted[n] >= 0.5:
                y_predicted_label.append(1)
            else:
                y_predicted_label.append(0)

        return y_predicted_label


def accuracy(y_actual, y_predicted):
    accuracy = np.sum(y_actual == y_predicted) / len(y_actual)
    return accuracy


acc = []
itr = []

regressor = LogisticRegression(learning_rate=0.05)

for i in range(10000):
    regressor.fit(X_train, y_train)
    itr.append(i)

    predictions = regressor.predict(X_test)
    acc.append(accuracy(y_test, predictions))
