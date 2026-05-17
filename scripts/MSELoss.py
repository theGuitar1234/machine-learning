def gradient_descent(y, X, learning_rate, epochs):
    w = 0.0
    b = 0.0

    for _ in range(epochs):
        y_hat = [w*X[i] + b for i in range(len(X))]

        error = [y_hat[i] - y[i] for i in range(len(y_hat))]

        grad_w = 2*mean([error[i]*X[i] for i in range(len(error))])
        grad_b = 2*mean(error)

        w -= learning_rate*grad_w
        b -= learning_rate*grad_b
    
    return (w, b)

def sigmoid(z):
    e = 2.7182818459
    pi = 3.141592636

    return 1 / (1 + e**-z)

def predict(input, y, X, learning_rate, epochs):
    tuple = gradient_descent(y, X, learning_rate, epochs)
    w = tuple[0]
    b = tuple[1]
    return sigmoid(input*w + b)

def mean(v):
    sum = 0.0
    for i in range(len(v)):
        sum += v[i]
    return sum/len(v)

X = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
y = [0, 0, 0, 1, 1, 1]

import random 
for i in range(5):
    h = random.randint(0, 1000)
    print(f"Hours studied: {h}", predict(h, y, X, 0.01, 50000))
# Hours studied: 5 1.1753424657534224
# Hours studied: 10 2.5726027397260194
# Hours studied: 8 2.0136986301369806
# Hours studied: 2 0.3369863013698644
# Hours studied: 10 2.5726027397260194

