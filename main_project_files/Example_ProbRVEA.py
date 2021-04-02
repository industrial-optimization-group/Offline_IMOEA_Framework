import sys
sys.path.insert(1, '/mnt/i/AmzNew/')
from desdeo_problem.Problem import DataProblem

from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from desdeo_emo.EAs.OfflineRVEA import ProbRVEAv3
from desdeo_emo.EAs.OfflineRVEA import RVEA

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygmo import non_dominated_front_2d as nd2

problem_name = "DTLZ2"
prob = test_problem_builder(name=problem_name, n_of_objectives=2, n_of_variables=10)

x = lhs(10, 100)
y = prob.evaluate(x)

#print(y)

x_names = [f'x{i}' for i in range(1,11)]
y_names = ["f1", "f2"]
print(x_names)

data = pd.DataFrame(np.hstack((x,y.objectives)), columns=x_names+y_names)
data_pareto = nd2(y.objectives)
y.objectives[data_pareto]
problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names)

problem.train(GaussianProcessRegressor, {"kernel": Matern(nu=3/2)})
#evolver_opt = ProbRVEAv3(problem, use_surrogates=True)
evolver_opt = RVEA(n_iterations=5, problem=problem, use_surrogates=True)
while evolver_opt.continue_evolution():
    evolver_opt.iterate()

print(evolver_opt.population.fitness)
print(evolver_opt.population.uncertainity)