import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x = b

c = [-10, -7, -5, -10, 0, 0, 0, 0, 0]
a = [
    [0, 0, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 1]
]
b = [9, 5, 7, 10, 15]

solution = scipy.optimize.linprog(c, A_eq = a, b_eq = b)

print(solution)
