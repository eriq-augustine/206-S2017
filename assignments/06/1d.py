import numpy

a = numpy.matrix([
    [8.9, 8.1, 3.5, 3.8],
    [9.6, 2.4, 8.3, 5.7],
    [5.5, 9.3, 5.9, 0.8],
    [1.4, 3.5, 5.5, 0.5],
    [1.5, 2.0, 9.2, 5.3],
    [2.6, 2.5, 2.9, 7.8],
    [8.4, 6.2, 7.6, 9.3],
    [2.5, 4.7, 7.5, 1.3]
])

b = numpy.matrix([58.6, 84.4, 46.9, 31.0, 67.5, 59.1, 100.6, 46.0]).T

e = numpy.matrix([1, 1, 1, 1])

f = 10.0

print((a.T * a).I * (a.T * b - 0.5 * (e.T * ((e * (a.T * a).I * e.T).I * (2.0 * (e * (a.T * a).I * a.T * b) - (2.0 * f))))))
