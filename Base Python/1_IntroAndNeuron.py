# Random values
inputs = [1.0 , 2.0 , 3.0 , 2.5] # Inputs from first layer
weights = [0.2 , 0.8 , -0.5 , 1.0]
bias = 2

# Hidden Layer Neuron output
# Sum(inputs * weights) + bias --> dot(inputs, weights) + bias
output =   inputs[0]*weights[0]\
         + inputs[1]*weights[1]\
         + inputs[2]*weights[2]\
         + inputs[3]*weights[3]\
         + bias
print(output)
