import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/Offline_IMOEA_Framework/')
#import beetle_objective
from desdeo_emo.EAs.RVEA_design_opt import ProbRVEAv3
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from desdeo_problem.Variable import variable_builder
from desdeo_problem.Objective import _ScalarObjective
from desdeo_problem.Objective import VectorObjective
from desdeo_problem.Problem import MOProblem
from desdeo_emo.EAs.RVEA_design_opt import ProbRVEAv3
from desdeo_emo.EAs.RVEA_design_opt import RVEA

def get_chol(X, beta):
    mu = expit(np.inner(X, beta))
    weight = np.sqrt(mu*(1 - mu))
    return X * weight[:, None]

def cost(x_1, beta):
    return np.sum(1 / expit(beta[0] + beta[1]*x_1))

def design_objectives(x1, x2, beta, n, obj_scale):
    X = np.empty([n, 4])
    X[:, 0] = np.ones(n)
    X[:, 1] = x2 # Note the order here, 1st column is the dose of the current phase, i.e., x2
    X[:, 2] = x1
    # NOTE, simplified interaction (either 1 or 0), so we can focus on the magnitude of beta[3]
    X[:, 3] = (X[:, 1]*X[:, 2] > 0).astype(float)
    Z = get_chol(X, beta)
    #print("Z=",Z)
    #y1 = obj_scale[0] * np.linalg.slogdet(np.matmul(Z.T, Z))[1] # log-scale might be more stable, sign is not needed
    y1 = obj_scale[0] * np.linalg.det(np.matmul(Z.T, Z)) # I changed this
    
    y2 = obj_scale[1] * cost(X[:, 1], beta)
    return np.array([y1, y2])

def objective_unc(x, 
                beta_mean=np.array([0.85, -0.5, -0.75, 0]),
                beta_sd=np.array([0.0, 0.0, 0.0, 0.25]),
                n_eval=100,
                obj_scale = (-1, 1),
                seed = 123):

    m = np.shape(x)[0]
    n = m // 2
    objs = np.empty([n_eval, 2])
    rng = np.random.default_rng(seed)
    beta_len = np.shape(beta_mean)[0]
    beta = np.empty([n_eval, beta_len])
    
    ### I changed this to beta 3 as uniform random. Normal distribution should be constrained

    #for j in range(beta_len):
    #    beta[:,j] = rng.normal(beta_mean[j], beta_sd[j], n_eval)
    
    for j in range(beta_len-1):
        beta[:,j] = np.repeat(beta_mean[j], n_eval)
    beta[:,3] = np.random.rand(n_eval)-0.5
    
    for i in range(n_eval):
        objs[i,:] = design_objectives(x[0:n], x[n:m], beta[i,:], n, obj_scale)
    
    #return objs
    #return np.array([np.mean(objs[:, 0]), np.std(objs[:, 0]), np.mean(objs[:, 1]), np.std(objs[:, 1])])
    #print([np.mean(objs[:, 0]), np.mean(objs[:, 1])])
    return np.array([np.mean(objs[:, 0]), np.mean(objs[:, 1])])

def objective_unc2(x, 
                beta_mean=np.array([0.85, -0.5, -0.75, 0]),
                beta_sd=np.array([0.0, 0.0, 0.0, 0.25]), 
                n_eval=100, 
                obj_scale = (-1, 1), 
                seed = 123):

    m = np.shape(x)[0]
    n = m // 2
    objs = np.empty([n_eval, 2])
    rng = np.random.default_rng(seed)
    beta_len = np.shape(beta_mean)[0]
    beta = np.empty([n_eval, beta_len])

    ### I changed this to beta 3 as uniform random. Normal distribution should be constrained

    #for j in range(beta_len):
    #    beta[:,j] = rng.normal(beta_mean[j], beta_sd[j], n_eval)
    
    for j in range(beta_len-1):
        beta[:,j] = np.repeat(beta_mean[j], n_eval)
    beta[:,3] = np.random.rand(n_eval)*0.25
    #beta[:,3] = np.ones(n_eval)*0.25
    
    for i in range(n_eval):
        objs[i,:] = design_objectives(x[0:n], x[n:m], beta[i,:], n, obj_scale)
    return objs


beta_mean=np.array([0.85, -0.5, -0.75, 0])
#beta_mean=np.array([5, -10, 0 , 0])
beta_sd = np.array([0.0, 0.0, 0.0, 0.25])
n_eval = 100
n_vars = 100
n_objs = 2
#def f_1(x):
#    return 0

#def f_2(x):
#    return 0

def f_xx(x):
    #x=np.random.logistic(1-0.5*(1-x))
    return list(map(objective_unc2, x))

def vect_f(x):
    #x=np.random.logistic(1-0.5*(1-x))
    if isinstance(x, list):
        if len(x) == n_vars:
            return [objective_unc(x)]
        elif len(x[0]) == n_vars:
            return list(map(objective_unc, x))
    else:
        if x.ndim == 1:
            return [objective_unc(x)]
        elif x.ndim == 2:
            return list(map(objective_unc, x))
    raise TypeError("Unforseen problem, contact developer")

x_names = [f'x{i}' for i in range(1,n_vars+1)]
y_names = [f'f{i}' for i in range(1,n_objs+1)]

list_vars = variable_builder(x_names,
                             initial_values = np.zeros(n_vars),
                             lower_bounds=np.zeros(n_vars),
                             upper_bounds=np.ones(n_vars))

f_objs = VectorObjective(name=y_names, evaluator=vect_f)
#f2 = _ScalarObjective(name='f2', evaluator=f_2)

problem = MOProblem(variables=list_vars, objectives=[f_objs])
evolver = ProbRVEAv3(design_model=f_xx, problem=problem, n_gen_per_iter=10, n_iterations=5)
#evolver = RVEA(design_model=f_xx, problem=problem, n_gen_per_iter=100, n_iterations=50)
while evolver.continue_evolution():
    evolver.iterate()
print("Solutions:",evolver.population.objectives)
