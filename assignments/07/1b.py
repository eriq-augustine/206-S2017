import conjugateGradient

import numpy

import math

# Note that we are subbing x in for q in the original problem.

def objective(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]

    return x1 * x1 + x2 * x2 + x3 * x3 \
        + x1 * x2 + (1.0 / 5.0 * x2 * x3) \
        + -10 * x1 + -20 * x2 + -30 * x3 \
        + math.exp(x1) + math.exp(2 * x2) + math.exp(3 * x3)

def gradient(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]

    # Derivative wrt x1
    dx1 = 2 * x1 + x2 - 10 + math.exp(x1)
    dx2 = 2 * x2 + x1 + (1.0 / 5.0 * x3) - 20 + 2 * math.exp(2 * x2)
    dx3 = 2 * x3 + (1.0 / 5.0 * x2) - 30 + 3 * math.exp(3 * x3)

    # Dimensions correct?
    return numpy.matrix([[dx1, dx2, dx3]])

def hessian(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]

    # Double Derivative wrt x1
    ddx1 = 2 + math.exp(x1)
    ddx2 = 2 + 4 * math.exp(2 * x2)
    ddx3 = 2 + 9 * math.exp(3 * x3)

    # Dimensions correct?
    return numpy.matrix([[ddx1, ddx2, ddx3]])

if __name__ == '__main__':
    objectiveLambda = lambda x: objective(x)
    gradientLambda = lambda x: gradient(x)
    hessianLambda = lambda x: hessian(x)

    guess = numpy.matrix([[10.0], [10.0], [10.0]])
    res = conjugateGradient.conjugateGradient(guess, objectiveLambda, gradientLambda, hessianLambda)
    x, objectiveValue, g, xHistory, objectiveHistory = res

    print(xHistory)
    print(objectiveHistory)
