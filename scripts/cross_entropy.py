import math

def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def cross_entropy_average(sum, input):
    return -1/len(input) * sum

def loss(y, input, w, b):
    sum = 0
    eps = 1e-12
    for i in range(len(X)):
        p_hat = sigmoid(linear_model(w, input[i], b))
        sum += y[i]*math.log(p_hat + eps) - (1 - y[i])*math.log(1 - p_hat + eps)
    return f"""
        For the weight {w} and bias {b}, 
        the accuracy of the linear model is: {cross_entropy_average(sum, input)}
    """

def linear_model(w, x, b):
    return w*x+b

X = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 10]
y = [0,   0,  0,   0, 0,   0, 1,   0, 1,   0,  1,   1,  1,   1,  1,   1,  1,   1,  1,  1]

print(loss(y, X, 1.4562492955153008, -5.448905251095365))
# For the weight 1.4562492955153008 and bias -5.448905251095365,
# the accuracy of the linear model is: -0.0015504892253229258