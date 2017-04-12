import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x <= b

c = [-10, -12, -30]
a = [
    [2, 0.5, 1],
    [1, 1.5, 2],
    [3, 4.0, 5]
]
b = [1500, 3000, 6000]

solution = scipy.optimize.linprog(c, a, b)

print("Bread: ", solution.x[0])
print("Cake: ", solution.x[1])
print("Muffins: ", solution.x[2])
