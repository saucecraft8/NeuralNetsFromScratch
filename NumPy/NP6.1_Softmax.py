import numpy as np

layer_outputs = [4.8 , 1.21 , 2.385]

# Exponential Function
exp_vals = np.exp(layer_outputs)

# Normalization
norm_vals = exp_vals / np.sum(exp_vals)

print(norm_vals)
print(sum(norm_vals))
