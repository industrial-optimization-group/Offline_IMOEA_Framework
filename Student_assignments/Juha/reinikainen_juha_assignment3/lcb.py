import numpy as np
from desdeo_emo.EAs.NSGAIII import NSGAIII

import problems

def optimize_with_lcb(the_real_problem, surrogates, X, y, max_evals):
    """
    Optimize multiobjective problem the_real_problem
    using surrogates with given initial data X,y
    terminates after max_evals amount of true evaluations of
    the_real_problem has been done.
    uses lower confidence bound to choose
    the points to evaluate
    """
    lcbProblem = problems.make_lcb(surrogates, the_real_problem, 1e-5)
    surrogateProblem = problems.make_surrogate(surrogates, the_real_problem)

    optimizer_acquisition = NSGAIII(
        lcbProblem, 10, n_iterations=1, n_gen_per_iter=10)
    optimizer = NSGAIII(surrogateProblem, 50,
                        n_iterations=1, n_gen_per_iter=10)

    # initial training of surrogates
    for i, surrogate in enumerate(surrogates):
        # each surrogate with its corresponding objective
        surrogate.fit(X, y[:, i])

    n_evals = 0
    while n_evals < max_evals:
        # while optimizer.continue_evolution():
        #     optimizer.iterate()
        # individuals, solutions = optimizer.end()

        # optimize the acquisition
        while optimizer_acquisition.continue_evolution():
            optimizer_acquisition.iterate()
        individuals, solutions = optimizer_acquisition.end()

        # evaluate chosen ones with real objective
        solutions = the_real_problem.evaluate(individuals).objectives
        n_evals += len(solutions)

        # append evaluated ones to dataset
        X = np.append(X, individuals, axis=0)
        y = np.append(y, solutions, axis=0)

        # retrain the surrogates
        for i, surrogate in enumerate(surrogates):
            surrogate.fit(X, y[:, i])

    # final solution
    while optimizer.continue_evolution():
        optimizer.iterate()
    individuals, solutions = optimizer.end()

    return individuals, solutions