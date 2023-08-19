import numpy as np
import nnfs

# Forcing conditions to match course (easier to check code)
np.random.seed(0)
nnfs.init()

def create_data(points , classes):
    X = np.zeros((points * classes , 2))
    y = np.zeros(points * classes , dtype = 'uint8')

    for i in range(classes):
        ix = range(points * i , points * (i + 1))
        r = np.linspace(0 , 1 , points) # radius
        t = np.linspace(i * 4 , i + 4 , points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r*np.sin(t*2.5) , r * np.cos(t * 2.5)]
        y[ix] = i

    return X , y

class Layer_Dense:
    def __init__(self , n_inputs , n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs , n_neurons)
        self.bias = np.zeros((1 , n_neurons))
        self.output = 0

    def forward(self , inputs):
        self.output = np.dot(inputs , self.weights) + self.bias
        return self.output # to have the option of storing/calling the output


class Activation_ReLU:
    def __init__(self):
        self.output = 0

    def forward(self , inputs):
        self.output = np.maximum(0 , inputs)
        return self.output

class Activation_Softmax:
    def __init__(self):
        self.output = 0

    def forward(self , inputs):
        exp_vals = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))
        self.output = exp_vals / np.sum(exp_vals , axis=1 , keepdims=True)
        return self.output

class Loss:
    def calculate(self , output , y):
        sample_losses = self.forward(output , y)
        data_loss = np.mean(sample_losses) # Batch Loss
        return data_loss

    # Abstract
    def forward(self , output , y):
        return output , y

class Loss_CategoricalCrossentropy(Loss):
    def forward(self , y_pred , y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred , 1e-7 , 1 - 1e-7)

        correct_confidences = 1e-7
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples) , y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true , axis = 1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
