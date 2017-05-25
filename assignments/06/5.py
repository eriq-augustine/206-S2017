import numpy

a = numpy.matrix([
    [-1, -6, 1, 3],
    [-1, -2, 7, 1],
    [0, 3, -10, -1],
    [-6, -11, -2, 12],
    [1, 6, -1, -3]
])
b = numpy.matrix([-3, 5, -8, -7, 4]).T
c = numpy.matrix([47, 93, 17, -93]).T
lagrange = numpy.matrix([3, 2, 2, 7, 0]).T

print(c.T + (lagrange.T * a))
