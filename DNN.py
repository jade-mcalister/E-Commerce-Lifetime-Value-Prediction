#import libraries
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error
import numpy as np
from Main import X_train, X_test, y_train, y_test

class DNN:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.bias = {}
        self.weights = {}

        # initialize layers

    #6 input neurons, 5 hidden neurons (h1), 3 hidden neurons, one output neuron
    input_neurons = X_train.shape[1]
    h1_neurons = 5
    h2_neurons = 3
    output_neuron = 1

    # sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # initialize random weights and biases
    learning_rate = 0.01

    weights_input_h1 = np.random.uniform(size=(input_neurons, h1_neurons))
    biases_input_h1 = np.random.uniform(size=(1, h1_neurons))

    weights_h1_h2 = np.random.uniform(size=(h1_neurons, h2_neurons))
    biases_h1_h2 = np.random.uniform(size=(1, h2_neurons))

    weights_h2_output = np.random.uniform(size=(h2_neurons, output_neuron))
    biases_h2_output = np.random.uniform(size=(1, output_neuron))

    # forward propagation
    def forward_pass(self, weights_input_h1, biases_input_h1, weights_h1_h2, biases_h1_h2, weights_h2_output,
                     biases_h2_output):
        h1_input = np.dot(X_train, weights_input_h1) + biases_input_h1  #y=wx+b using inputs
        h1_output = self.sigmoid(h1_input)  # implement sigmoid function

        h2_input = np.dot(h1_output, weights_h1_h2) + biases_h1_h2 #using output from h1
        h2_output = self.sigmoid(h2_input)  #sigmoid

        output_layer_input = np.dot(h2_output, weights_h2_output) + biases_h2_output  #using output from h2
        output = self.sigmoid(output_layer_input)  # implement sigmoid function

