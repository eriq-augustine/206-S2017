import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x = b

c = [1, 6, -7, 1, 5]
# We are abusing list building notation to easily get the correct number of zeros.
a = [
    [5, -4, 13, -2, 1],
    [1, -1, 5, -1, 1],
]
b = [20, 8]

solution = scipy.optimize.linprog(c, A_eq = a, b_eq = b)

print(solution)
