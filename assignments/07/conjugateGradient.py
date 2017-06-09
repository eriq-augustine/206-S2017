import math
import numpy

CONVERGENCE_DIFF = 1e-15
GRADIENT_CUTOFF = 1e-9

# All matric work, input, and output will be done with numpy.matrix.
# Returns: [x, objectiveValue, g, xHistory, objectiveHistory]
def conjugateGradient(initialX, objective, gradient, hessian):
    # If the initial values will overflow, then just take the log of them.
    try:
        _ = (initialX.T * initialX)[0, 0]
        for row in range(initialX.shape[0]):
            for col in range(initialX.shape[1]):
                _ += math.exp(initialX[row, col])
    except OverflowError:
        for row in range(initialX.shape[0]):
            for col in range(initialX.shape[1]):
                initialX[row, col] = math.log(initialX[row, col])

    x = initialX

    oldObjectiveValue = None
    objectiveValue = objective(x)

    # Keep track of some historical values.
    xHistory = [initialX.copy()]
    objectiveHistory = [objectiveValue]

    while (oldObjectiveValue == None or abs(objectiveValue - oldObjectiveValue) > CONVERGENCE_DIFF):
        # Compute Hessian to use for next n = 2 steps
        # (This is the "Q" of the quadratic approximation)
        H = hessian(x)

        # Compute gradient at the center of the quadratic approximation
        g_center = gradient(x).T

        # This finds the "b" term of the quadratic approximation
        b_lin = H * x - g_center

        # Loop through n = 2 steps
        for n in range(2):
            # Compute gradient from quadratic approximation
            g = H * x - b_lin

            if n == 0:
                # First step should be a gradient step
                d = -g
            else:
                gamma = (g.T * H * d)[0, 0] / (d.T * H * d)[0, 0]
                d = -(g - gamma * d)

            # If gradient miniscule, avoid dividing by something close to zero
            if (numpy.linalg.norm(g) > GRADIENT_CUTOFF):
                # Choose step size to minimize quadratic approximation:
                # alpha = -g.T * d / (d.T * H * d)

                # Uncomment below to choose step size using Newton's method
                alpha = linesearch(x, d, gradient, hessian)
            else:
                alpha = 0

            # Update x
            x += alpha * d

            oldObjectiveValue = objectiveValue
            objectiveValue = objective(x)

            # Log histories
            xHistory.append(x.copy())
            objectiveHistory.append(objectiveValue)

    g = gradient(x).T

    return [
        x,
        objectiveValue,
        g,
        xHistory,
        objectiveHistory
    ]

def l2Norm(vector):
    sum = 0
    for value in vector:
        sum += value * value

    return math.sqrt(sum)

def linesearch(x, d, gradient, hessian):
    # This is a Newton's method line search
    alpha = 0
    for i in range(100):
        alpha += -(gradient(x + alpha * d) * d)[0, 0] / (d.T * hessian(x + alpha * d) * d)[0, 0]

    return alpha
