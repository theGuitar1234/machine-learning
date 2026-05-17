import numpy as np
import math

# def sigmoid(z):
#     e = 2.7182818285
#     #print(1 / (1 + np.exp(-z)))
#     return 1.0 / (1.0 + e**-z)

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def gradient_descent(y, X, learning_rate, epochs):
    w = 0.0
    b = 0.0

    for _ in range(epochs):
        y_hat = [sigmoid(w*X[i] + b) for i in range(len(X))]

        error = [y_hat[i] - y[i] for i in range(len(y_hat))]

        grad_w = mean([error[i]*X[i] for i in range(len(error))])
        grad_b = mean(error) 

        w -= learning_rate*grad_w
        b -= learning_rate*grad_b
    
    print(w, b)
    
    return (w, b)

def predict(input, tuple):
    # tuple = gradient_descent(y, X, learning_rate, epochs)
    w = tuple[0]
    b = tuple[1]
    
    # return input*w + b
    return sigmoid(input*w + b)
    # if (sigmoid(input*w + b) > 0.5):
    #     return "The student will pass"
    # else:
    #     return "The student will fail"

def mean(v):
    sum = 0.0
    for i in range(len(v)):
        sum += v[i]
    return sum/len(v)

# # Hours studied
# X = [1, 2, 4, 5, 7] 
# # Actual scores (y)
# y = [50, 55, 70, 75, 85]

# X = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
# y = [0, 0, 0, 1, 1, 1]

import random 

X = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10]
y = [0,   0,  0,   0, 0,   0, 1,   0, 1,   0,  1,   1,  1,   1,  1,   1,  1,   1,  1,  1]

if __name__ == "__main__":
    tuple = gradient_descent(y, X, 0.01, 50000)
    for i in range(5):
        h = random.randint(0, 10)
        print(f"Hours studied: {h}", predict(h, tuple))
        # Hours studied: 9 0.9995276597013458
        # Hours studied: 4 0.5929301794798776
        # Hours studied: 5 0.862040397364535
        # Hours studied: 4 0.5929301794798776
        # Hours studied: 1 0.0181163858897732
