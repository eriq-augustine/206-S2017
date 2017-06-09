import math
import numpy

ZERO_CUTOFF = 1e-12
GRADIENT_CUTOFF = 1e-9

# Working set is a list of booleans indicating what constraints (rows in A) are currently active.
# Returns: [x, objectiveValue]
# TODO(eriq): Generalize
def gradientProjection(A, b, x, workingSet, objective, gradient, hessian):
    # If the initial values will overflow, then just take the log of them.
    try:
        _ = (x.T * x)[0, 0]
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                _ += math.exp(x[row, col])
    except OverflowError:
        for row in range(x.shape[0]):
            for col in range(x.shape[1]):
                x[row, col] = math.log(x[row, col])

    # Compute the working A (AB)
    AB = computeAB(A, workingSet)

    # Keep going until are lambda's are positive.
    while (True):
        # Stop when d is sufficiently small.
        while (True):
            g = gradient(x)

            if (AB.shape[0] == 0 or AB.shape[1] == 0):
                d = -g.T
            else:
                d = -(numpy.eye(AB.shape[0]) - AB.T * (AB * AB.T).I * AB) * g.T

            if (numpy.linalg.norm(d) <= ZERO_CUTOFF):
                break

            # If gradient miniscule, avoid dividing by something close to zero
            if (numpy.linalg.norm(g) > GRADIENT_CUTOFF):
                # Use Newton's method
                alpha = linesearch(x, d, gradient, hessian)
            else:
                alpha = 0.0

            # The furthest we would move.
            tempX = x + alpha * d

            # Check if we would violate any constraints (check for positive components).
            constraintCheck = (A * tempX - b).tolist()

            # Smallest alpha amongst all the violoated constraints.
            minAlpha = alpha
            minAlphaConstraint = None

            for i in range(len(constraintCheck)):
                # No violation
                if (constraintCheck[i][0] <= 0):
                    continue

                # Violation
                violatedAlpha = (d.I * ((A[i]).I * b[i] - x))[0, 0]

                if (minAlphaConstraint == None or minAlpha < violatedAlpha):
                    minAlpha = violatedAlpha
                    minAlphaConstraint = i
                    
            # Add any violoated constraint to the working set.
            if (minAlphaConstraint != None):
                workingSet[minAlphaConstraint] = True

            AB = computeAB(A, workingSet)

            # If alpha is small enough, then just bail.
            if (abs(minAlpha) < ZERO_CUTOFF):
                break

            # Update x
            x += minAlpha * d

        rawLambdaValues = (-((AB * AB.T).I * AB * g.T)).tolist()

        # Augment the lambda values so that the indicies match A and workingSet.
        lambdaValues = []
        rawIndex = 0
        for inWorkingSet in workingSet:
            if (inWorkingSet):
                lambdaValues.append(rawLambdaValues[rawIndex][0])
                rawIndex += 1
            else:
                lambdaValues.append(0)

        done = True
        for i in range(len(lambdaValues)):
            if (lambdaValues[i] >= 0):
                continue

            # This constraint must be removed.
            workingSet[i] = False

        if (done):
            return x, objective(x)

        AB = computeAB(A, workingSet)

def computeAB(A, workingSet):
    A = A.tolist()
    AB = []

    for i in range(len(workingSet)):
        if (workingSet[i]):
            AB.append(A[i])

    return numpy.matrix(AB)

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
