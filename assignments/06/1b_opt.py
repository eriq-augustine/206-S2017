import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x = b

# x, S(+/-)
c = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# We are abusing list building notation to easily get the correct number of zeros.
a = [
    [8.9, 8.1, 3.5, 3.8, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [9.6, 2.4, 8.3, 5.7, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [5.5, 9.3, 5.9, 0.8, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1.4, 3.5, 5.5, 0.5, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1.5, 2.0, 9.2, 5.3, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
    [2.6, 2.5, 2.9, 7.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
    [8.4, 6.2, 7.6, 9.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
    [2.5, 4.7, 7.5, 1.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
]
b = [58.6, 84.4, 46.9, 31.0, 67.5, 59.1, 100.6, 46.0]

solution = scipy.optimize.linprog(c, A_eq = a, b_eq = b)

print(solution)