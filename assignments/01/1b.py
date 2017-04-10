import scipy.optimize

# Minimize: c^T * x
# Subject to: A_ub * x <= b_ub

c = [-10, -12, -30]
a = [
    [2, 0.5, 1],
    [1, 1.5, 2],
    [3, 4, 5]
]
b = [1500, 3000, 6000]

print(scipy.optimize.linprog(c, a, b))
