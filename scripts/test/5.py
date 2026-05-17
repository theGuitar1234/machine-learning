import numpy as np

x = np.array([20, 60, 1])       # [hours, problems, sleep]
w = np.array([2.0, 0.5, 1.0])  # weights
b = 10.0                       # bias

y = np.dot(w, x) + b

print(y)
