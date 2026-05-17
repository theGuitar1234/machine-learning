def sigmoid(z):
    e = 2.7182818459
    return 1 / (1 + e**-z)

def linear_model(x, w, b):
    return x*w + b

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
    
    return (w, b)

def mean(v):
    sum = 0.0
    for i in range(len(v)):
        sum += v[i]
    return sum/len(v)

X = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10]
y = [0,   0,  0,   0, 0,   0, 1,   0, 1,   0,  1,   1,  1,   1,  1,   1,  1,   1,  1,  1]

import random 

if __name__ == "__main__":
    tuple = gradient_descent(y, X, 0.01, 50000)
    for i in range(5):
        h = random.randint(0, 10)
        print(f"Hours studied: {h}", sigmoid(linear_model(h, tuple[0], tuple[1])))