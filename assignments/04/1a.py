import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x = b

c = [-7, 7, -2, -1, -6, 0]
# We are abusing list building notation to easily get the correct number of zeros.
a = [
    [+3, -1, +1, -2, +0, +0],
    [+2, +1, +0, +1, +1, +0],
    [-1, +3, +0, -3, +0, +1],
]
b = [-3, 4, 12]

solution = scipy.optimize.linprog(c, A_eq = a, b_eq = b)

print(solution)
