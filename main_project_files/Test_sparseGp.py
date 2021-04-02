import GPy
import numpy as np
from desdeo_problem.Problem import DataProblem

from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from desdeo_emo.EAs.OfflineRVEA import ProbRVEAv3

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygmo import non_dominated_front_2d as nd2


problem_name = "DTLZ4"
nvars = 10
nobjs = 2
nsamples = 5000
prob = test_problem_builder(problem_name, nvars, nobjs)

x = lhs(nvars, nsamples)
y = prob.evaluate(x)

x_names = [f'x{i}' for i in range(1,nvars)]
y_names = ["f1", "f2"]


k = GPy.kern.RBF(1)
m_full = GPy.models.GPRegression(x,y[])
m_full.optimize('bfgs')
m_full.plot()
print(m_full)


