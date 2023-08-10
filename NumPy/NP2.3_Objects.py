"""
Normalizing data to 1 (-1 to 1 or 0 to 1) because data can scale exponentially
through the model as more layers and therefore more calculations are done

Biases tend to be initialized at 0
However, biases can 'zero out' resulting in a zero being output throughout
the network if input starts at 0 (one solution: initialize bias as non-zero)
Ex:
Input: 0
Layer1: 0 * weight + 0 == 0
Layer2: 0 * weight + 0 == 0
...
Output: 0
"""
import numpy as np
from Abstracting import Layer_Dense as Dense

# Random values
# 3 Samples
X = [[1.0 , 2.0 , 3.0 , 2.5],
     [2.0 , 5.0 , -1.0 , 2.0],
     [-1.5 , 2.7 , 3.3 , -0.8]]

# Layer 1
weights = [[0.2 , 0.8 , -0.5 , 1.0],
           [0.5 , -0.91 , 0.26 , -0.5],
           [-0.26 , -0.27 , 0.17 , 0.87]]
bias = [2 , 3 , 0.5]

# Layer 2
weights2 = [[0.1 , -0.14 , 0.5],
            [-0.5 , 0.12 , -0.33],
            [-0.44 , 0.73 , -0.13]]
bias2 = [-1 , 2 , -0.5]

layer1 = Dense(4 , 5)
layer2 = Dense(5 , 2)
output = layer1.forward(X)
output = layer2.forward(output)
print(output)
