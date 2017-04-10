import scipy.optimize

# Minimize: c^T * x
# Subject to: A_ub * x <= b_ub

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

print(scipy.optimize.linprog(c, a, b))
