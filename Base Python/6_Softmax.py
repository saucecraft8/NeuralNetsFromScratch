from math import e

layer_outputs = [4.8 , 1.21 , 2.385]

# Exponential Function
exp_vals = [e**x for x in layer_outputs]

# Normalization
norm_base = sum(exp_vals)
norm_vals = [x / norm_base for x in exp_vals]

print(norm_vals)
print(sum(norm_vals))
