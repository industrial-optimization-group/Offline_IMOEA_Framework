#%%
import numpy as np
from desdeo_problem.Problem import DataProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
from desdeo_problem.Problem import DataProblem
import pandas as pd
from main_project_files.surrogate_fullGP import FullGPRegressor as fgp
from main_project_files.surrogate_sparseGP import SparseGPRegressor as sgp
from desdeo_emo.EAs.OfflineRVEA import RVEA

import plotly.graph_objects as go
from desdeo_emo.othertools.plotlyanimate import animate_init_, animate_next_

problem_name = "DTLZ2"
nvars = 10
nobjs = 2
nsamples = 100
prob = test_problem_builder(problem_name, nvars, nobjs)

x = lhs(nvars, nsamples)
x_test = lhs(nvars, int(nsamples*0.2))

y = prob.evaluate(x)
#y=np.asarray(y[0])

y_test = prob.evaluate(x_test)
y_test=np.asarray(y_test[0])

x_names = [f'x{i}' for i in range(1,nvars+1)]
y_names = ["f1", "f2"]

data = pd.DataFrame(np.hstack((x,y.objectives)), columns=x_names+y_names)
problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names)

problem.train(fgp)
#%%
evolver_opt = RVEA(problem, use_surrogates=True)
while evolver_opt.continue_evolution():
    evolver_opt.iterate()

front_true = evolver_opt.population.objectives
print(front_true)

individuals = evolver_opt.population.individuals
# %%
objectives = evolver_opt.population.objectives
fig1 = go.Figure(data=go.Scatter(x=objectives[:,0],y=objectives[:,1], mode="markers"))
fig1



# %%
objectives

# %%
problem.evaluate(x, True)

# %%
