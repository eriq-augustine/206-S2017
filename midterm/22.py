import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x = b

c = [-3, -2, -3, 0, 0]
# We are abusing list building notation to easily get the correct number of zeros.
a = [
    [2, 3, 4, 1, 0],
    [4, 3, 2, 0, 1],
]
b = [5, 6]

solution = scipy.optimize.linprog(c, A_eq = a, b_eq = b)

print(solution)
