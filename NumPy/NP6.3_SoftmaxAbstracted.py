from Abstracting import Layer_Dense , Activation_ReLU , Activation_Softmax
from nnfs.datasets import spiral_data

X , y = spiral_data(samples = 100 , classes = 3)

dense1 = Layer_Dense(2 , 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3 , 3)
activation2 = Activation_Softmax()

x = dense1.forward(X)
x = activation1.forward(x)
x = dense2.forward(x)
x = activation2.forward(x)
print(x)
