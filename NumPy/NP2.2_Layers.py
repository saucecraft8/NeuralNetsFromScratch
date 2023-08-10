import numpy as np

# Random values
inputs = [[1.0 , 2.0 , 3.0 , 2.5],
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
layer1_output = np.dot(inputs , np.transpose(np.array(weights))) + bias
layer2_output = np.dot(layer1_output , np.transpose(np.array(weights2))) + bias2
print(layer2_output)
