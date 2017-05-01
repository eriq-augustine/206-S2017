import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x = b

c = [2, 3, 2, 2]
# We are abusing list building notation to easily get the correct number of zeros.
a = [
    [+1, +2, +1, +2],
    [+1, +1, +2, +4],
]
b = [+8, +7]

solution = scipy.optimize.linprog(c, A_eq = a, b_eq = b)

print(solution)
