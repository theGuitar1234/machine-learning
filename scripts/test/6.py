import numpy as np
from sklearn.linear_model import LinearRegression

# Our simple dataset
# Hours studied (x)
X = np.array([[1],
              [2],
              [4],
              [5],
              [7]])

# Actual scores (y)
y = np.array([50, 55, 70, 75, 85])

# Create the model
model = LinearRegression()

# Fit (learn w and b)
model.fit(X, y)

# Inspect learned parameters
w = model.coef_      # slope(s)
b = model.intercept_ # bias

print("Learned w:", w)
print("Learned b:", b)

# Predict for a new student who studied 3 hours
x_new = np.array([[3]])
y_pred = model.predict(x_new)
print("Predicted score for 3 hours:", y_pred)
