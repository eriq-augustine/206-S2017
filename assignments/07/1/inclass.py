import conjugateGradient

import numpy

import math

def objective(x, Q, b):
    return 0.5 * (x.T * Q * x)[0, 0] - (b.T * x)[0, 0] + math.exp(x[0, 0] / 10.0) + math.exp(2.0 * x[1, 0] / 10.0)

def gradient(x, Q, b):
    val = numpy.matrix([
        math.exp(x[0, 0] / 10.0),
        2.0 * math.exp(2.0 * x[1, 0] / 10.0)
    ])

    return x.T * Q - b.T + (1.0 / 10.0) * val

def hessian(x, Q):
    val = numpy.matrix([
        [math.exp(x[0, 0] / 10.0), 0.0],
        [0.0, 4.0 * math.exp(2.0 * x[1, 0] / 10.0)]
    ])

    return Q + ((1.0 / (100.0)) * val)

if __name__ == '__main__':
    Q = numpy.matrix([
        [2, 1],
        [1, 2]
    ])

    b = numpy.matrix([
        [10],
        [10]
    ])

    objectiveLambda = lambda x: objective(x, Q, b)
    gradientLambda = lambda x: gradient(x, Q, b)
    hessianLambda = lambda x: hessian(x, Q)

    guess = numpy.matrix([[10.0], [10.0]])
    # guess = numpy.matrix([[1000000.0], [1000000.0]])
    res = conjugateGradient.conjugateGradient(guess, objectiveLambda, gradientLambda, hessianLambda)
    x, objectiveValue, g, xHistory, objectiveHistory = res

    print(xHistory)
    print(objectiveHistory)
