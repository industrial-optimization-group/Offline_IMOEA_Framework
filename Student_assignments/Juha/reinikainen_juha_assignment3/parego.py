import numpy as np
from desdeo_tools.utilities.fast_non_dominated_sorting import non_dominated
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

import itertools
import math

import problems

class ParEGO:
    """
    ParEGO algorithm
    Optimize multiobjective problem the_real_problem
    using surrogate trained on
    scalarized objectives with given initial data X,y
    terminates after max_evals amount of true evaluations of
    the_real_problem has been done.
    """
    def __init__(self):
        pass

    def createWeightVectors(self, M, H):
        """
        Creates evenly distributed weight vectors
        in M dimensional space with H fractions 
        """
        #0/H,1/H,...H/H
        fractions = np.linspace(0, 1, H+1)
        N = math.comb(H+M-1, M-1)
        U = np.zeros((N, M))
        ui = 0
        #iterate possible fraction sets and pick those that
        #add up to 1
        for u in itertools.product(fractions, repeat=M):
            if math.isclose(sum(u), 1.0):
                U[ui] = np.array(u)
                ui += 1
        return U

    def scalarize(self, F, w):
        """
        Augmented Tchebycheff function 
        """
        WF = w * F
        WFmax = WF.max(axis=1)
        p = 0.05
        WFsum = WF.sum(axis=1)
        Fscalarized = WFmax + p * WFsum
        return Fscalarized.reshape(-1, 1)

    def newLambda(self, W):
        """
        get uniformly randomly selected weight vector
        """
        i = np.random.randint(0, len(W))
        return W[i]

    def optimize(self, the_real_problem, surrogate, X, y, max_evals, s, seed):
        """
        s: controls how many weight vectors are generated
        """
        optimizer_acquisition = GA(pop_size=100)
        optimizer = GA(pop_size=100)

        W = self.createWeightVectors(y.shape[1], s)

        surrogateProblem = problems.Surrogate(the_real_problem, surrogate)

        for _ in range(max_evals):
            w = self.newLambda(W)
            F = self.scalarize(y, w)
            surrogate.fit(X, F)

            best = minimize(surrogateProblem, optimizer, ("n_gen", 10), seed=seed)

            # optimize the acquisition
            ei = problems.ExpectedImprovement(the_real_problem, surrogate, best.F)
            res = minimize(ei, optimizer_acquisition, ("n_gen", 10), seed=seed)
            x_t = res.X

            #evaluate with real
            y_t = the_real_problem.evaluate(x_t)
            # append evaluated ones to dataset
            X = np.append(X, [x_t], axis=0)
            y = np.append(y, [y_t], axis=0)

        #return nondominated solutions in X
        non_dominated_ones = non_dominated(y)
        return X[non_dominated_ones], y[non_dominated_ones]