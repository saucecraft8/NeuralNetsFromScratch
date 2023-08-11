import numpy as np

layer_outputs = [[4.8 , 1.21 , 2.385],
                 [8.9 , -1.81 , 0.2],
                 [1.41 , 1.05 , 0.026]]

# Exponential Function
exp_vals = np.exp(layer_outputs)

# Normalization
# axis=0 --> columns, axis=1 --> rows
norm_vals = exp_vals / np.sum(exp_vals , axis=1 , keepdims=True)

print(norm_vals)
