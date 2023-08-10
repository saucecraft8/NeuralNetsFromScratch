"""
Batch refers to how many samples the model can 'see' in one step when training
More samples == more information at a time allowing it to better generalize
to the data set since it has more information and therefore a better idea of
what the end product will look like

Ex:
it's easier to draw a linear approximation using 8 points than 1

Negative:
Training with batch sizes too large or batch sizes that encompass the entire
dataset will generalize the model to that specific dataset (overfitting)

Common Batch Sizes: 32-64
"""
import numpy as np

# Random values
inputs = [[1.0 , 2.0 , 3.0 , 2.5],
          [2.0 , 5.0 , -1.0 , 2.0],
          [-1.5 , 2.7 , 3.3 , -0.8]]

weights = [[0.2 , 0.8 , -0.5 , 1.0],
           [0.5 , -0.91 , 0.26 , -0.5],
           [-0.26 , -0.27 , 0.17 , 0.87],]

bias = [2 , 3 , 0.5]

# Currently 3x4 * 3x4
# output = np.dot(weights , inputs) + bias

"""
Dot-Product of 2 matrices is just multiplying the matrices (ORDER MATTERS)
Remember same number rule

CAN multiply matrices if dimensions are:
2x1 * 1x3 == 2x3 ; 3x4 * 4x3 == 3x3 ; 5x5 * 5x8 == 5x8
CANNOT multiply matrices if dimensions are:
1x2 * 3x1 ; 3x4 * 3x4 ; 1x2 * 3x4
"""
# np.array().T --> Transpose
output = np.dot(inputs , np.transpose(np.array(weights))) + bias
print(output)
