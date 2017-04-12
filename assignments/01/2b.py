import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x <= b

c = [-5, -5, -10, -10, -7, -7]
a = [
    [1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1, 0],
    [1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1]
]
b = [10, 15, 5, 7, 8, 12]

solution = scipy.optimize.linprog(c, a, b)

for i in range(len(solution.x)):
    print("x%d: %f" % (i + 1, solution.x[i]))
