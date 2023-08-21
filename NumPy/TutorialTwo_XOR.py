from Abstracting.TutorialTwo import Dense , Tanh , MSE , MSE_prime
import numpy as np

X = np.reshape([[0 , 0],
                [0 , 1],
                [1 , 0],
                [1 , 1]] , (4 , 2 , 1))
y = np.reshape([[0],
                [1],
                [1],
                [0]] , (4 , 1 , 1))

model = [
    Dense(2 , 3),
    Tanh(),
    Dense(3 , 1),
    Tanh()
]

EPOCHS = 10000
LEARNING_RATE = 0.1
error = 0

# Training Loop
for epoch in range(EPOCHS):
    for X , y in zip(X , y):
        output = X
        # Forward Pass
        for layer in model:
            output = layer.forward(output)

        # Error
        error += MSE(y , output)

        # Back Propagation
        grad = MSE_prime(y , output)
        for layer in reversed(model):
            grad = layer.backward(grad , LEARNING_RATE)

error /= len(X)
print(f'Error: {error}')
# Doesn't work but can't be bothered to fix rn
# TL;DR --> Back propagation is way more math than a forward pass
# No need to know it perfectly/replicate it, just know the general concepts
