import scipy.optimize

# Minimize: c^T * x
# Subject to: a * x = b

# x = [S1+, S1-, ..., S9+, S9-, P1, P2, P3, P4]
c = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
# We are abusing list building notation to easily get the correct number of zeros.
a = [
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 154, 209,   0,   0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 357,   0,   0, 109],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 505,   0, 253,  30],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 303, 103,  54, 106],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 602,  52,   0,  56],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 581,   0, 207,  30],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 432,   0,  52,   0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 359, 500, 652,  41],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 100, 150, 230,  10]
]

b = [154, 427, 505, 337, 602, 589, 432, 359, 175]

solution = scipy.optimize.linprog(c, A_eq = a, b_eq = b)
if (solution.status == 2):
    print("Program is infeasible")
else:
    print(solution)

    pValues = solution.x[18:]
    for i in range(len(pValues)):
        print("p%d = %3.2f" % (i + 1, pValues[i]))
