import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x <= b

c = [-30, -20, -40, -25, -10, 0, 0]
a = [
    [2, 1, 3, 3, 1, 1, 0],
    [3, 2, 2, 1, 1, 0, 1]
]
b = [700, 1000]

solution = scipy.optimize.linprog(c, a, b)

print(solution)

parts = solution.x[:5]
for i in range(len(parts)):
    print("Part %d - %.2f (100 units)" % (i + 1, parts[i]))
