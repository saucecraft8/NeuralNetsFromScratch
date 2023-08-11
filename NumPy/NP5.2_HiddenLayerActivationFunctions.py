import numpy as np
from nnfs.datasets import spiral_data
from Abstracting import Layer_Dense
from Abstracting import Activation_ReLU

X , y = spiral_data(100 , 3)

layer1 = Layer_Dense(2 , 5)
activation1 = Activation_ReLU()

# Optimizer SHOULD fix zeroing out (consider initializing biases as non-zero)
output = layer1.forward(X)
output = activation1.forward(output)
print(output)