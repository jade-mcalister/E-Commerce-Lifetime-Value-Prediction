import pandas as pd
import numpy as np
from Main import X_train, X_test, y_train, y_test

class DNN:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.bias = {}
        self.weights = {}

        # initialize layers

    #6 input neurons
    input_neurons = X_train.shape[1]
    hidden_neurons = 4
    output_neuron = 1

    # sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))