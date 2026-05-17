
def linear_model(w, x, b):
    return dot_product(w, x)+b

def dot_product(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

def relu(z):
    return max(0, z)

def sigmoid(z):
    e = 2.7182818459
    return 1 / (1 + e**-z)

def soft_max(Z):   
    e = 2.7182818459
    result = [0 for _ in range(len(Z))] 
    sum = 0
    m = max(Z)
    for i in range(len(Z)):
        sum += e**(Z[i] - m)
    for i in range(len(Z)):
        result[i] = e**(Z[i] - m) / sum
    return result

X = [
    [[0.2, 1.1]],
    [[0.4, 0.9]],
    [[1.2, 0.1]],
    [[1.0, 0.2]],
    [[2.0, 1.5]],
    [[2.2, 1.7]],
]

import random 

def initWeightsAndBias(r, c):
    W = [[random.uniform(-1, 1) for _ in range(c)] for _ in range(r)]
    b = [random.uniform(-1, 1) for _ in range(c)]
    return W, b
        
def mul(A, B):
    if not A or not B:
        raise ValueError("Empty matrix.")
    if any(len(row) != len(A[0]) for row in A):
        raise ValueError("Matrix A is not rectangular.")
    if any(len(row) != len(B[0]) for row in B):
        raise ValueError("Matrix B is not rectangular.")

    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])

    if n != n2:
        raise ValueError(f"Incompatible shapes: A is {m}x{n}, B is {n2}x{p}")

    result = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            total = 0
            for k in range(n):
                total += A[i][k] * B[k][j]
            result[i][j] = total

    return result

def add(mat1, b):
    return [mat1[0][i] + b[i] for i in range(len(b))]

def pretty_print(m):
    for i in m:
        print(i)

def forward_pass(X, W, b, act):
    result = []
    for i in X:
        logits = add(mul(i, W), b)
        if act is not None:
            logits = [act(v) for v in logits]
        result.append([logits])
    return result

n_in = len(X[0][0])
n_out = 3

WB1 = initWeightsAndBias(2, 3)
WB2 = initWeightsAndBias(3, 4)
WB3 = initWeightsAndBias(4, 2)

WB = [WB1, WB2, WB3]

def initLayers(X, hidden_layers, output_dim=None, start_width=None):
    n_in = len(X[0][0])

    if start_width is None:
        start_width = n_in

    WB = []
    width = start_width

    for _ in range(hidden_layers):
        n_out = width
        WB.append(initWeightsAndBias(n_in, n_out))
        n_in = n_out
        width *= 2
    
    if output_dim is not None:
        WB.append(initWeightsAndBias(n_in, output_dim))
    
    return WB

def neural_network(X, WB, act):
    H = X
    for i in range(len(WB)):
        if (i == len(WB) - 1):
            Z = forward_pass(H, WB[i][0], WB[i][1], None)
            probs = []
            for j in Z:
                probs.append(soft_max(j[0]))
            return probs
        H = forward_pass(H, WB[i][0], WB[i][1], act)

num_layers = 10

# print(initLayers(X, 4, 2, 2))
print(neural_network(X, initLayers(X, 4, 2, 3), relu))