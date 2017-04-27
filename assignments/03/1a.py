import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x = b

c = [-3, 1, 3, -1]
# We are abusing list building notation to easily get the correct number of zeros.
a = [
    [+1, +2, -1, +1],
    [+2, -2, +3, +3],
    [+1, -1, +2, -1],
]
b = [0, 9, 6]

solution = scipy.optimize.linprog(c, A_eq = a, b_eq = b)

print(solution)
