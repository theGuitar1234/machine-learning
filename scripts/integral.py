def f(x):
    return x**2

def bajillion_integral(f, a, n, dx):
	area = 0.0
	for i in range(n):
		x = a + i * dx         
		area += f(x) * dx
	return area

def trapezoid_integral(f, a, n, dx): 
	area = 0.0
	for i in range(n):
		x0 = a + i * dx
		x1 = x0 + dx
		area += (f(x0) + f(x1)) * 0.5 * dx
	return area
    
a = 1
b = 5
n = 10**5
dx = (b - a) / n

print(bajillion_integral(f, a, n, dx)) #41.33285333440035
print(trapezoid_integral(f, a, n, dx))