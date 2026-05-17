import calculus.logistic_regression_matrixDONE as lrm

def train_one_vs_rest(X, y_multiclass, K, learning_rate, epochs, lrm):
    params = []
    for c in range(K):
        y_binary = [1 if yi == c else 0 for yi in y_multiclass]
        print(y_binary)
        w, b = lrm.gradient_descent(y_binary, X, learning_rate, epochs)
        params.append((w, b))
    return params 

def predict_proba_softmax(x, params, lrm):
    logits = [lrm.linear_model(w, x, b) for (w, b) in params]
    return soft_max(logits)

def predict_class(x, params, lrm):
    p = predict_proba_softmax(x, params, lrm)
    return max(range(len(p)), key=lambda k: p[k])

def soft_max(Z):   
    e = 2.7182818459
    result = [0 for _ in range(len(Z))] 
    sum = 0
    for i in range(len(Z)):
        sum += e**Z[i]
    for i in range(len(Z)):
        result[i] = e**Z[i] / sum
    return result

Dog = [
    [1.0, 1.0],
    [3.0, 10.0],
    [4.0, 10.0],
    [5.0, 25.0]
]

Cat = [
    [1.0, 5.0],
    [2.0, 10.0],
    [4.0, 20.0],
    [5.0, 25.0]
]

Rabbit = [
    [1.0, 5.0],
    [2.0, 10.0],
    [4.0, 24.0],
    [6.0, 25.0]
]



X_all = Dog + Cat + Rabbit

y_all = [0]*len(Dog) + [1]*len(Cat) + [2]*len(Rabbit)

K = 3

import calculus.logistic_regression_matrixDONE as lrm

params = train_one_vs_rest(X_all, y_all, K=3, learning_rate=0.0001, epochs=50000, lrm=lrm)

probs = [predict_proba_softmax(x, params, lrm) for x in X_all]

print("max prob anywhere:", max(max(row) for row in probs))
print("first sample probs:", probs[0])
print("first sample predicted class:", predict_class(X_all[0], params, lrm))
# max prob anywhere: 0.7835887281353794
# first sample probs: [0.5865183993389201, 0.20931299309763285, 0.20416860756344704]
# first sample predicted class: 0

X_test = [
    [2.5,  8.0],
    [4.5, 12.0],
    [5.5, 22.0],     
    [1.2,  2.0],
    [3.8, 24.5],     
    [6.5, 26.0],      
]

for x in X_test:
    p = predict_proba_softmax(x, params, lrm)   # [P(Dog), P(Cat), P(Rabbit)]
    pred = predict_class(x, params, lrm)        # 0, 1, or 2
    print(f"x={x}  probs={p}  pred={pred}")

