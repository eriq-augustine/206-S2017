import scipy.optimize

ATTACKING_STRATS = ['AA', 'BB', 'CC', 'AB', 'AC', 'BC']
DEFENDING_STRATS = ['AAA', 'BBB', 'CCC', 'AAB', 'AAC', 'BBA', 'BBC', 'CCA', 'CCB', 'ABC']
BASES = ['A', 'B', 'C']

def createPayoffs(baseValues):
    payoffs = []

    for attackingStrat in ATTACKING_STRATS:
        row = []

        for defendingStrat in DEFENDING_STRATS:
            score = 0
            for baseIndex in range(len(BASES)):
                attackCount = attackingStrat.count(BASES[baseIndex])

                if (attackCount == 0):
                    continue

                # Attackers - Defenders
                attackAdvantage = attackingStrat.count(BASES[baseIndex]) - defendingStrat.count(BASES[baseIndex])

                if (attackAdvantage > 0):
                    score += baseValues[baseIndex]
                elif (attackAdvantage == 0):
                    score += baseValues[baseIndex] / 2.0

            row.append(score)

        payoffs.append(row)

    return payoffs

# min r
# st. 1(10x1)^T * x = 1
#     Mx <= r * 1(6x1)
#     x >= 0
#
# Tranformed into
# min c^T * x
# st. A_eq * x = b_eq
#     st. Ax <= b
#     x >= 0
def solve(baseValues):
    # We need to transform some of the values to get them into the proper c^T * x, st. A * x <= b form.
    # r will be appended onto x.

    A_eq = [[1] * len(DEFENDING_STRATS)]
    A_eq[0].append(0)
    b_eq = 1

    # The damage (r) is moving into the payoff matrix, so b will be all zero.
    b = [0] * len(ATTACKING_STRATS)

    A = createPayoffs(baseValues)
    # Add -1 to each row of A (M).
    for i in range(len(A)):
        A[i].append(-1.0)

    # r is the last value 
    c = [0] * len(DEFENDING_STRATS)
    c.append(1)

    solution = scipy.optimize.linprog(c = c, A_ub = A, b_ub = b, A_eq = A_eq, b_eq = b_eq)

    # print(solution)
    print("Base Values: ", ', '.join(["%s: %d" % (BASES[i], baseValues[i]) for i in range(len(BASES))]))
    print("Game Value: ", solution.fun)
    print("Defender Probabilities:")
    for i in range(len(DEFENDING_STRATS)):
        print("   ", DEFENDING_STRATS[i], ": ", solution.x[i])

    # Use the slack to calcualate the attacker probabilites.
    sum = 0.0
    for slackValue in solution.slack:
        sum += slackValue

    print("Attacker Probabilities:")
    for i in range(len(ATTACKING_STRATS)):
        print("   ", ATTACKING_STRATS[i], ": ", (solution.slack[i] / sum))

if __name__ == '__main__':
    # print(createPayoffs([1, 1, 1]))

    print("Part b")
    solve([1, 1, 1])

    print("Part c")
    solve([1, 2, 3])

    print("Part d")
    solve([1, 1, 50])
