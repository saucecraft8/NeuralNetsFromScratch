"""
Normalizing data to 1 (-1 to 1 or 0 to 1) because data can scale exponentially
through the model as more layers and therefore more calculations are done
"""
import numpy as np

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

# np.array().T --> Transpose
layer1_output = np.dot(X , np.transpose(np.array(weights))) + bias
layer2_output = np.dot(layer1_output , np.transpose(np.array(weights2))) + bias2
print(layer2_output)


