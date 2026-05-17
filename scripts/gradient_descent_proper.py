import torch
from torch import nn

class Optimizer(nn.Module):
        def __init__(self, lr, epochs):
                super().__init__()
                self.lr = lr
                self.epochs = epochs
        def gdc(self, X, Y, w=0.0, b=0.0):
                for _ in range(self.epochs):
                        grads = [(X[i]*w + b) - Y[i] for i in range(len(Y))]
                        dW = [grads[i]*X[i] for i in range(len(X))]
                        delta = sum(dW)/len(dW)
                        delta_db = sum(grads)/len(grads)
                        w = w - self.lr*delta
                        b = b - self.lr*delta_db
                return (w, b)

class Regression(nn.Module):
        def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.randn(-1, 1))
                self.b = nn.Parameter(0.0)
        def linear_model(self, X):
                return X @  self.w + self.b

coefficient = 4.5
bias = 2.7
limit = 5
X = [i for i in range(limit)]
Y = [X[j]*coefficient + bias for j in range(limit)]
w, b = Optimizer(0.1, 1000).gdc(X, Y)
print(w, b)