import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Forward Propagation
    # Abstract
    def forward(self , input):
        pass

    # Back Propagation
    # Abstract
    def backward(self , output_gradient , learning_rate):
        pass

class Dense(Layer):
    def __init__(self , input_size , output_size):
        super().__init__()
        self.weights = np.random.randn(output_size , input_size)
        self.bias = np.random.randn(output_size , 1)

    def forward(self , input):
        self.input = input
        return np.dot(self.weights , self.input) + self.bias

    def backward(self , output_gradient , learning_rate):
        weights_gradient = np.dot(output_gradient , learning_rate)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return np.dot(self.weights.transpose() , output_gradient)

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime # Derivative of activation

    def forward(self , input):
        self.input = input
        return self.activation(self.input)

    def backward(self , output_gradient , learning_rate):
        return np.multiply(output_gradient , self.activation_prime(self.input))

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2 # Derivative of tanh
        super().__init__(tanh , tanh_prime)

# Mean Squared Error
def MSE(y_true , y_pred):
    return np.mean(np.power(y_true - y_pred , 2))

# Derivative of Mean Squared Error
def MSE_prime(y_true , y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)