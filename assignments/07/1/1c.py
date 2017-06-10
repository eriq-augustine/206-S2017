import gradientProjection

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

    # Derivative
    dx1 = 2 * x1 + x2 - 10 + math.exp(x1)
    dx2 = 2 * x2 + x1 + (1.0 / 5.0 * x3) - 20 + 2 * math.exp(2 * x2)
    dx3 = 2 * x3 + (1.0 / 5.0 * x2) - 30 + 3 * math.exp(3 * x3)

    return numpy.matrix([[dx1, dx2, dx3]])

def hessian(x):
    x1 = x[0, 0]
    x2 = x[1, 0]
    x3 = x[2, 0]

    # Double Derivative
    dx1dx1 = 2 + math.exp(x1)
    dx1dx2 = 1.0
    dx1dx3 = 0

    dx2dx1 = -1.0
    dx2dx2 = 2 + 4 * math.exp(2 * x2)
    dx2dx3 = 0.2

    dx3dx1 = 0
    dx3dx2 = 0.2
    dx3dx3 = 2 + 9 * math.exp(3 * x3)

    return numpy.matrix([
        [dx1dx1, dx1dx2, dx1dx3],
        [dx2dx1, dx2dx2, dx2dx3],
        [dx3dx1, dx3dx2, dx3dx3],
    ])

if __name__ == '__main__':
    objectiveLambda = lambda x: objective(x)
    gradientLambda = lambda x: gradient(x)
    hessianLambda = lambda x: hessian(x)

    A = numpy.matrix([
        [0, 1, 0],
        [0, 0, 1]
    ])

    b = numpy.matrix([
        [2],
        [0.5]
    ])

    workingSet = [False, False]
    x = numpy.matrix([
        [0.0],
        [0.0],
        [0.0]
    ])

    x, objectiveValue = gradientProjection.gradientProjection(A, b, x, workingSet, objective, gradient, hessian)

    print("Solution:")
    print(x)
    print(objectiveValue)
