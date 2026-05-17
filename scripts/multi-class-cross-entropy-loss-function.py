import math
import calculus.logistic_regression_matrixDONE as lmr

def soft_max(Z):   
    e = 2.7182818459
    result = [0 for _ in range(len(Z))] 
    sum = 0
    for i in range(len(Z)):
        sum += e**Z[i]
    for i in range(len(Z)):
        result[i] = e**Z[i] / sum
    return result

def cross_entropy_average(sum, input):
    return -1/len(input) * sum

def multiclass_cross_entropy(y, X, W, b):
    total = 0.0
    eps = 1e-12
    N = len(X)

    for i in range(N):
        logits = logits_multiclass(W, X[i], b)
        p = soft_max(logits)
        total += sum(y[i][k] * math.log(p[k] + eps) for k in range(len(p)))

    return -total / N

def cross_entropy_index(y, X, W, b, eps=1e-12):
    total = 0.0
    N = len(X)

    for i in range(N):
        p = soft_max(logits_multiclass(W, b, X[i]))
        total += math.log(p[y[i]] + eps)

    return -total / N

def logits_multiclass(W, b, x):
    return [dot_product(Wk, x) + bk for Wk, bk in zip(W, b)]

def dot_product(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total   

X = [
    [0.2, 1.1],
    [0.4, 0.9],
    [1.2, 0.1],
    [1.0, 0.2],
    [2.0, 1.5],
    [2.2, 1.7],
]

y = [0, 0, 1, 1, 2, 2]

y = [
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [0,0,1],
    [0,0,1],
    [0,0,1],
]

W = [
    [-1.0,  1.2],
    [ 0.8, -0.4],
    [ 1.1,  0.2],
]

b = [ 1.0, 0.0, -1.0]

print(cross_entropy_index(y, X, W, b))
#1.3431756173620357