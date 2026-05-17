import math

class Vector:

    def __init__(self, coords):
        self.coords = coords

    def scale(self, factor):
        for i in range(len(self.coords)):
            self.coords[i] = self.coords[i]*factor

    def add(self, factor):
        factor + [0 for _ in range(abs(len(self.coords) - len(factor)))]
        self.coords += [0 for _ in range(abs(len(self.coords) - len(factor)))]
        for i in range(len(self.coords)):
            self.coords[i] = self.coords[i] + factor[i]

    def dot_product(self, factor):
        factor + [0 for _ in range(abs(len(self.coords) - len(factor)))]
        coords = self.coords + [0 for _ in range(abs(len(self.coords) - len(factor)))]
        sum = 0
        for i in range(len(coords)):
            sum += coords[i]*factor[i]
        return sum
    
    def dot_product_cos(self, factor, alpha):
        factor + [0 for _ in range(abs(len(self.coords) - len(factor)))]
        coords = self.coords + [0 for _ in range(abs(len(self.coords) - len(factor)))]
        return self.lngth(factor) * self.lngth(coords) * math.cos(alpha)
    
    def length(self, vector):
        sum = 0
        for i in vector.coords:
            sum += i**2
        return sum**0.5
    
    def lngth(self, coords): 
        sum = 0
        for i in coords:
            sum += i**2
        return sum**0.5
        
    def __str__(self):
        return str(self.coords)
    
    def distance(self, factor):
        factor + [0 for _ in range(abs(len(self.coords) - len(factor)))]
        coords = self.coords + [0 for _ in range(abs(len(self.coords) - len(factor)))]
        return self.lngth(factor) - self.lngth(coords)

coords = [1, 2]
vector = Vector(coords)

vector.scale(3)
print(vector)

factor = [1, 2, 3]
# vector.add(factor)
print(vector.dot_product_cos(factor, 60))
# print(vector.length())
# print(vector)