import numpy as np
import nnfs

from Abstracting import Layer_Dense , Activation_ReLU , Activation_Softmax , Loss_CategoricalCrossentropy
from nnfs.datasets import vertical_data

nnfs.init()
X , y = vertical_data(samples = 100 , classes = 3)

dense1 = Layer_Dense(2 , 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3 , 3)
activation2 = Activation_Softmax()
loss_func = Loss_CategoricalCrossentropy()

lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.bias.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.bias.copy()

# 10000 Epochs
for i in range(10000):

    # Really should create getter/setter methods
    dense1.weights += 0.05 * np.random.randn(2 , 3) # Remember randn is based on Normal distribution
    dense1.bias += 0.05 * np.random.randn(1 , 3)
    dense2.weights += 0.05 * np.random.randn(3 , 3) # Matches above parameters
    dense2.bias += 0.05 * np.random.randn(1 , 3)

    x = dense1.forward(X)
    x = activation1.forward(x)
    x = dense2.forward(x)
    x = activation2.forward(x)

    loss = loss_func.calculate(x , y)
    pred = np.argmax(x , axis = 1)
    acc = np.mean(pred == y)

    if loss < lowest_loss:
        print(f'New weights found\nEpoch: {i} | Loss: {loss} | Accuracy: {acc}')
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.bias.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.bias.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.bias = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.bias = best_dense2_biases.copy()
