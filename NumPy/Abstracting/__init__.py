import numpy as np

np.random.seed(0)

class Layer_Dense:
    def __init__(self , n_inputs , n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs , n_neurons)
        self.bias = np.zeros((1 , n_neurons))
        self.output = 0

    def forward(self , inputs):
        self.output = np.dot(inputs , self.weights) + self.bias
        return self.output # to have the option of storing/calling the output
