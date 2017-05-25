import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x = b

# x, S(+/-)
c = [47, 93, 17, -93, 0, 0, 0, 0, 0]
# We are abusing list building notation to easily get the correct number of zeros.
a = [
    [-1, -6, 1, 3, 1, 0, 0, 0, 0],
    [-1, -2, 7, 1, 0, 1, 0, 0, 0],
    [0, 3, -10, -1, 0, 0, 1, 0, 0],
    [-6, -11, -2, 12, 0, 0, 0, 1, 0],
    [1, 6, -1, -3, 0, 0, 0, 0, 1]
]
b = [-3, 5, -8, -7, 4]

solution = scipy.optimize.linprog(c, A_eq = a, b_eq = b)

print(solution)
