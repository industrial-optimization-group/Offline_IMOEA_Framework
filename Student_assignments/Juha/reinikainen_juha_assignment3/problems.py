import numpy as np
from scipy.stats import norm
from desdeo_problem import MOProblem, VectorObjective
from pymoo.core.problem import Problem

"""
Wrappers and implementations of problems for single and multiobjective optimization
for pymoo and desdeo interfaces
"""

class ExpectedImprovement(Problem):
    def __init__(self, p: Problem, model, fxb, xi = 1e-5):
        """
        p: problem object
        model: model object
        fxb: best objective function value found
        xi: a parameter controlling the degree of exploration
        """
        super().__init__(n_var=p.n_var, n_obj=1,
                         n_constr=0, xl=p.xl, xu=p.xu)
        self.model = model
        self.fxb = fxb
        self.xi = xi

    def _evaluate(self, X, out, *args, **kwargs):
        x_mean, x_std = self.model.predict(X, return_std=True)
        x_var = np.sqrt(x_std).reshape(-1, 1)
        # expected improvement
        inner = (self.fxb - x_mean - self.xi)/x_var
        left = (self.fxb - x_mean - self.xi) * norm.cdf(inner)
        right = x_var * norm.pdf(inner)
        EI = left + right
        # negate to fit the interface requiring minimization
        out["F"] = -EI

class Surrogate(Problem):
    """
    Surrogate to Problem wrapper
    """
    def __init__(self, p, model):
        """
        p: problem object
        model: surrogate model
        """
        super().__init__(n_var=p.n_var, n_obj=1,
                         n_constr=0, xl=p.xl, xu=p.xu)
        self.model = model

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self.model.predict(X)

def make_surrogate(models, problem: MOProblem):
    """
    Problem with evaluation using models
    """
    def comp(X):
        #function values
        y = np.zeros((len(X), problem.n_of_objectives))
        for modelI, model in enumerate(models):
            x_mean = model.predict(X, return_std=False)
            y[:, modelI] = x_mean
        return y

    obj_names = [f"f{i}" for i in range(len(models))]
    objectives = VectorObjective(obj_names, comp)
    return MOProblem([objectives], problem.variables)



def make_lcb(models, problem: MOProblem, beta):
    """
    Create multiobjective problem with evaluation 
    with models on lcb with given beta value
    """
    def comp_lcb(X):
        lcbs = np.zeros((len(X), len(models)))
        for modelI, model in enumerate(models):
            x_mean, x_std = model.predict(X, return_std=True)
            x_var = np.sqrt(x_std)
            lcb = x_mean - beta * x_var
            lcbs[:, modelI] = lcb
        return lcbs

    obj_names = [f"f{i}" for i in range(len(models))]
    objectives = VectorObjective(obj_names, comp_lcb, maximize=False)
    return MOProblem([objectives], problem.variables)