import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/Offline_IMOEA_Framework/')

from desdeo_problem.Problem import DataProblem

from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from desdeo_emo.EAs.OfflineRVEA import ProbRVEAv3
import os
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main_project_files.surrogate_fullGP import FullGPRegressor as fgp
#from main_project_files.surrogate_sparseGP import SparseGPRegressor as sgp
#from main_project_files.surrogate_sparseGP_2 import SparseGPRegressor as sgp2
from desdeo_emo.EAs.OfflineRVEA import RVEA
#from pygmo import non_dominated_front_2d as nd2
from other_tools.non_domx import ndx
import scipy.io
from sklearn.neighbors import NearestNeighbors
import time
import GPy
from BIOMA_framework import interactive_optimize
#from ADM_Run import 
import pickle

save_model=True
max_samples = 50
max_iters = 5
gen_per_iter=100
data_folder = '/home/amrzr/Work/Codes/data'
init_folder = data_folder + '/AM_Samples_1000'
def read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run):
    mat = scipy.io.loadmat(init_folder+'/Initial_Population_' + problem_testbench + '_' + sampling +
                        '_AM_' + str(nvars) + '_'+str(nsamples)+'.mat')
    x = ((mat['Initial_Population_'+problem_testbench])[0][run])[0]
    if problem_testbench == 'DDMOPP':
        mat = scipy.io.loadmat(init_folder+'/Obj_vals_DDMOPP_'+sampling+'_AM_'+problem_name+'_'
                                + str(nobjs) + '_' + str(nvars)
                                + '_'+str(nsamples)+'.mat')
        y = ((mat['Obj_vals_DDMOPP'])[0][run])[0]
    elif problem_testbench == 'DTLZ':
        prob = test_problem_builder(
                    name=problem_name, n_of_objectives=nobjs, n_of_variables=nvars
                )
        y = prob.evaluate(x)[0]
    elif problem_testbench == 'GAA':
        mat = scipy.io.loadmat(init_folder+'/Obj_vals_GAA_'+sampling+'_AM_'+problem_name+'_'
                                + str(nobjs) + '_' + str(nvars)
                                + '_'+str(nsamples)+'.mat')
        y = ((mat['Obj_vals_GAA'])[0][run])[0]
    return x, y

def build_surrogates(problem_name, nobjs, nvars, nsamples, is_data, x_data, y_data, approach, problem_testbench):
    x_names = [f'x{i}' for i in range(1,nvars+1)]
    y_names = [f'f{i}' for i in range(1,nobjs+1)]
    row_names = ['lower_bound','upper_bound']
    if is_data is False:
        prob = test_problem_builder(problem_name, nvars, nobjs)
        x = lhs(nvars, nsamples)
        y = prob.evaluate(x)
        data = pd.DataFrame(np.hstack((x,y.objectives)), columns=x_names+y_names)
    else:
        data = pd.DataFrame(np.hstack((x_data,y_data)), columns=x_names+y_names)
    if problem_testbench == 'DDMOPP':
        x_low = np.ones(nvars)*-1
        x_high = np.ones(nvars)
    elif problem_testbench == 'DTLZ':
        x_low = np.ones(nvars)*0
        x_high = np.ones(nvars)
    elif problem_testbench == 'GAA':
        x_low = [0.240, 7.000, 0.000, 5.500, 19.000, 85.000, 14.000, 3.000, 0.460,
                    0.240, 7.000, 0.000, 5.500, 19.000, 85.000, 14.000, 3.000, 0.460,
                    0.240, 7.000, 0.000, 5.500, 19.000, 85.000, 14.000, 3.000, 0.460]

        x_high = [0.480, 11.000, 6.000, 5.968, 25.000, 110.000, 20.000, 3.750, 1.000,
                  0.480, 11.000, 6.000, 5.968, 25.000, 110.000, 20.000, 3.750, 1.000,
                  0.480, 11.000, 6.000, 5.968, 25.000, 110.000, 20.000, 3.750, 1.000]   
    bounds = pd.DataFrame(np.vstack((x_low,x_high)), columns=x_names, index=row_names)
    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names,bounds=bounds)
    problem.train(GaussianProcessRegressor, {"kernel": Matern(nu=3/2)})
    return problem

def run_optimizer(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, approach, run, path):
    if is_data is True:
        path_to_model = path+'_model'
        if os.path.exists(path_to_model) is False:
            x, y = read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run)
            surrogate_problem = build_surrogates(problem_name, nobjs, nvars, nsamples, is_data, x, y, approach, problem_testbench)
            if save_model is True:  
                outfile = open(path_to_model, 'wb')
                pickle.dump(surrogate_problem, outfile)
                outfile.close()  
        else:
            print("Using saved models...")
            infile = open(path_to_model, 'rb')
            surrogate_problem=pickle.load(infile)
            infile.close()  
    if approach == "interactive_uncertainty_new":
        population = run_optimizer_approach_interactive_new(surrogate_problem, path)
    elif approach == "adm_tests":
        #population = run_adm(surrogate_problem)
        print("To be done!")
    else:
        population = run_optimizer_approach_interactive(surrogate_problem, path)
    results_dict = {
            'individual_archive': population.individuals_archive,
            'objectives_archive': population.objectives_archive,
            'uncertainty_archive': population.uncertainty_archive,
            'individuals_solutions': population.individuals,
            'obj_solutions': population.objectives,
            'uncertainty_solutions': population.uncertainity                
        }
    return results_dict

def run_optimizer_approach_interactive_new(problem, path):
    print("Optimizing...")
    evolver_opt = interactive_optimize(problem, gen_per_iter, max_iters, path)
    return evolver_opt.population

def run_optimizer_approach_interactive(problem, path):
    print("Optimizing...")
    evolver_opt = interactive_optimize(problem, gen_per_iter, max_iters, path)
    return evolver_opt.population
