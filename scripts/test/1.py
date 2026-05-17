
def summation_i_squared(n):
    sum = 0
    for i in range (n+1):
        sum += i**2
    return sum

n = 5
print(summation_i_squared(n))