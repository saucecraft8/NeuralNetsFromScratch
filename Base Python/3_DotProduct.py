# Random values
inputs = [1.0 , 2.0 , 3.0 , 2.5] # Inputs from first layer
weights = [[0.2 , 0.8 , -0.5 , 1.0],
           [0.5 , -0.91 , 0.26 , -0.5], # Changed to -0.91
           [-0.26 , -0.27 , 0.17 , 0.87]]
bias = [2 , 3 , 0.5]

# Takes Dot Product of input with each list in weights and adds bias
def get_output(inputs , weights , bias):
    if len(inputs) != len(weights[0]) or len(bias) != len(weights):
        return None

    output = []
    for n_weights , n_bias in zip(weights, bias):
        total = 0
        for n_input , n_weight in zip(inputs , n_weights):
            total += n_input * n_weight
        output.append(total + n_bias)

    return output


# Hidden Layer Neuron output
output = get_output(inputs , weights , bias)
print(output)
