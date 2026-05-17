# def derivative(func, x) :
#     current = x
#     h = 1
#     while True :
#         next = (func(1, 0, 0, current+h) - func(1, 0, 0, current)) / h
#         h /= 2
#         if (abs(next-current) < 1e-5) :
#             return current
#         current = next

import math 

def quadratic(a, b, c, x) :
    return a*x**2 + b*x + c

def euler(x) :
    return math.pow(math.e, x)

def equation(x) :
    return 4*x**3

def tan(x) :
    return math.tan(x)

def absolute(x) :
    return abs(x)

def f(x):
    return math.cos(x)

def sinus(x) :
    return math.sin(x)

def func(x) :
    return math.exp(x)
# print(derivative(quadratic(a, b, c, x), x))
#lim(h->0) (f(x+h) - f(x)) / h
#x^2

def func(x):
    return x**2

current = 10
prev = None
h = 1

while True : 
    next = (func(current+h) - func(current)) / h
    if (prev is not None and abs(next - prev) < 1e-5) :
        break
    prev = next
    h /= 2

print(next)


