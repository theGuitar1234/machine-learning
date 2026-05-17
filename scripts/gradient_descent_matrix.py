def gradient_descent(y, X, learning_rate, epochs):
    w = [0.0] * len(X[0])
    b = 0.0

    for _ in range(epochs):
        y_hat = [dot_product(w, X[i]) + b for i in range(len(X))]

        error = [y_hat[i] - y[i] for i in range(len(y_hat))]

        grad_w = [0.0] * len(X[0])

        for j in range(len(X[0])):
            s = 0.0
            for i in range(len(X)):
                s += error[i] * X[i][j]
            grad_w[j] = (2.0 / len(X)) * s

        grad_b = (2.0 / len(X)) * sum(error)

        for j in range(len(X[0])):
            w[j] -= learning_rate * grad_w[j]
        b -= learning_rate * grad_b
    
    return f"""
        Predicted weight: {w}
        Predicted bias: {b}
    """

def dot_product(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total   

X = [
    [1, 5],
    [2, 10],
    [4, 20],
    [5, 25]
]

y = [50, 55, 70, 75]

print(gradient_descent(y, X, 0.0001, 50000))