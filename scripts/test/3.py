import numpy as np

# Two students: [hours_studied, problems_solved]
a = np.array([10, 50])
b = np.array([5, 20])

# Length (norm) of each
length_a = np.linalg.norm(a)
length_b = np.linalg.norm(b)

# Distance between them
diff = b - a
distance_ab = np.linalg.norm(diff)

print("Length of a:", length_a)
print("Length of b:", length_b)
print("Distance between a and b:", distance_ab)
