train_data = [50, 55, 70, 75, 85]
input = [
    [1],
    [2],
    [4],
    [5],
    [6]
]

def gradient_descent(train_data, input, learning_rate):
    w = 1000

    for _ in range(1000):
        grad = 2*(w*input - train_data) * input
        w = w - learning_rate*grad
    
    return w

def loss(train_data, input, w, b):
    sum = 0
    for i in range(len(train_data)):
        sum += (linear_model(w, input[i][0], b) - train_data[i])**2
    return f"""
        For the weight {w} and bias {b}, 
        the accuracy of the linear model is: {sum/len(input)}
    """

def linear_model(w, x, b):
    return w*x+b

print(gradient_descent(6, 3, 0.1))

w = 6.00877193
b = 45.166666666666664

print(loss(train_data, input, w, b))
# For the weight 6.00877193 and bias 44.166666666666664,
# the accuracy of the linear model is: 5.629039701908306