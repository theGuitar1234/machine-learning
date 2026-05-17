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

def predict_raw(input, y, X, learning_rate, epochs):
    tuple = gradient_descent(y, X, learning_rate, epochs)
    w = tuple[0]
    b = tuple[1]
    
    return input*w + b

def mean(v):
    sum = 0.0
    for i in range(len(v)):
        sum += v[i]
    return sum/len(v)

# Hours studied
X = [1, 2, 4, 5, 7] 
# Actual scores (y)
y = [50, 55, 70, 75, 85]

import random 

for i in range(5):
    h = random.randint(0, 10)
    print(f"Hours studied: {h}", predict_raw(h, y, X, 0.001, 50000))
# Hours studied: 9 The student will pass
# Hours studied: 99 The student will pass
# Hours studied: 94 The student will pass
# Hours studied: 76 The student will pass
# Hours studied: 29 The student will pass