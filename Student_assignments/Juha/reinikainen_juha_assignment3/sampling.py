from scipy.stats import qmc
from desdeo_problem import MOProblem
from pymoo.core.problem import Problem

from typing import Union

def create_data(problem: Union[MOProblem,Problem], n_samples, seed):
    """
    Create samples with latinhypercube sampling
    """
    if isinstance(problem, MOProblem):
        n_var = problem.n_of_variables
        lb = problem.get_variable_lower_bounds()
        ub = problem.get_variable_upper_bounds()
    else:
        n_var = problem.n_var
        lb = problem.xl
        ub = problem.xu
    # generate initial sample
    lhs = qmc.LatinHypercube(n_var, seed=seed)
    X = lhs.random(n_samples)
    X = qmc.scale(X, lb,ub)
    if isinstance(problem, MOProblem):
        y = problem.evaluate(X).objectives
        return X, y
    y = problem.evaluate(X)
    return X, y