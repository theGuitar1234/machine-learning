import math

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)
    
def gradient_descent(y, X, learning_rate, epochs):
    w = [0.0] * len(X[0])
    b = 0.0

    for _ in range(epochs):
        y_hat = [sigmoid(dot_product(w, X[i]) + b) for i in range(len(X))]

        error = [y_hat[i] - y[i] for i in range(len(y_hat))]

        grad_w = [0.0] * len(X[0])

        for j in range(len(X[0])):
            s = 0.0
            for i in range(len(X)):
                s += error[i] * X[i][j]
            grad_w[j] = s / len(X)

        grad_b = sum(error) / len(X)

        for j in range(len(X[0])):
            w[j] -= learning_rate * grad_w[j]
        b -= learning_rate * grad_b

    return (w, b)
    
    # return f"""
    #     Predicted weight: {w}
    #     Predicted bias: {b}
    # """

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def dot_product(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total   

def transpoze(mat):
    result = [[0 for _ in range(len(mat))] for _ in range(len(mat[0]))]
    for i in range(len(result)):
        for j in range(len(result[0])):
            result[i][j] = mat[j][i]
    return result

def predict(input, tuple):
    w = tuple[0]
    b = tuple[1]
    
    return sigmoid(input*w + b)

def linear_model(w, x, b):
    return dot_product(w, x)+b

X = [
    [1, 5],
    [2, 10],
    [4, 20],
    [5, 25]
]

y = [0, 0, 1, 1]

if __name__ == "__main__":
    print(gradient_descent(y, X, 0.0001, 50000))
    # Predicted weight: [0.021005091660459385, 0.105025458302298]
    # Predicted bias: -0.8787072565494798