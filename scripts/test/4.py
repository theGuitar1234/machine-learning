import numpy as np

# Features: [hours_studied, problems_solved, lectures_attended]
x = np.array([10, 40, 3])

# Weights: importance of each feature
w = np.array([0.5, 0.3, 0.2])

score = np.dot(w, x)

print(score)
