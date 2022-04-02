import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/Offline_IMOEA_Framework/')

from desdeo_problem.Problem import DataProblem
import pickle
from desdeo_problem.testproblems.TestProblems import test_problem_builder
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main_project_files.surrogate_fullGP import FullGPRegressor as fgp
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.ensemble import RandomForestClassifier

#from pygmo import non_dominated_front_2d as nd2
from other_tools.non_domx import ndx
import scipy.io
from sklearn.neighbors import NearestNeighbors
import time
import GPy
#from BIOMA_framework import interactive_optimize
from BIOMA_framework_worst import interactive_optimize
from BIOMA_framework_worst import full_optimize
import copy


FE_max = 500
max_iters = 10 #50
gen_per_iter= 100 #10
is_interact = False
read_saved_models = False

selection_type_dict = {'genericRVEA':'generic',
                        'probRVEA':'prob_only',
                        'probRVEA_constraint_v1':'prob_class_v1',
                        'probRVEA_constraint_v2':'prob_class_v2'}

def run_optimizer(problem_testbench, init_folder, nvars, nobjs, problem_spec, approach, run):
    model_file = problem_spec+'_'+str(run)+'_model'
    if read_saved_models:
        data_file = init_folder + '/test_runs/constrained_models' + model_file
        infile = open(data_file, 'rb')
        results_data = pickle.load(infile)
        infile.close()
        surrogate_problem = results_data["surrogate_problem"]
        print("loaded saved models...")
    else:
        surrogate_problem = build_surrogates(problem_testbench, init_folder, nvars, nobjs, problem_spec, run)
    classification_model = build_classification_failed(init_folder, problem_testbench, nvars, problem_spec, run)
    print(approach)
    selection_type = selection_type_dict[approach]
    print("Selection_type:",selection_type)
    population = run_optimizer_approach_full(surrogate_problem, classification_model, selection_type)
    results_dict = {
        'individual_archive': population.individuals_archive,
        'objectives_archive': population.objectives_archive,
        'uncertainty_archive': population.uncertainty_archive,
        'individuals_solutions': population.individuals,
        'obj_solutions': population.objectives,
        'uncertainty_solutions': population.uncertainity                
        }

    if read_saved_models is False:
        surrogate_problem_copy = copy.deepcopy(surrogate_problem)
        #classification_model_copy = copy.deepcopy(classification_model)
        models_dict = {
                'surrogate_problem': surrogate_problem_copy
                #'classification_model': classification_model_copy           
            }

        path = init_folder + '/constrained_models/' + problem_testbench
        outfile = open(path+'/' + model_file, 'wb')
        pickle.dump(models_dict, outfile)
        outfile.close()
    return results_dict


def build_surrogates(problem_testbench, init_folder, nvars, nobjs, problem_spec, run):
    print("Building surrogates...")
    file_success = init_folder + '/' + problem_testbench + '/' + problem_spec + '_' + str(run) + '_data_success.csv'    
    data_success = pd.read_csv(file_success)    
    x_data_success = data_success.values[:,0:nvars]
    y_data_success = data_success.values[:,nvars:nvars+nobjs]
    x_names = [f'x{i}' for i in range(1,nvars+1)]
    y_names = [f'f{i}' for i in range(1,nobjs+1)]
    row_names = ['lower_bound','upper_bound']
    data = pd.DataFrame(np.hstack((x_data_success,y_data_success)), columns=x_names+y_names)
    if problem_testbench == 'DBMOPP':
        x_low = np.ones(nvars)*-1
        x_high = np.ones(nvars)
    elif problem_testbench == 'DTLZ':
        x_low = np.ones(nvars)*0
        x_high = np.ones(nvars)
    bounds = pd.DataFrame(np.vstack((x_low,x_high)), columns=x_names, index=row_names)
    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names,bounds=bounds)
    problem.train(fgp)
    print("Surrogates built...")
    return problem


def build_classification_failed(init_folder, problem_testbench, nvars, problem_spec, run):
    print("Building classification surrogates...")
    file_class = init_folder + '/' + problem_testbench + '/' + problem_spec + '_' + str(run) + '_data_class.csv'
    data_class = pd.read_csv(file_class)   
    x_class=data_class.values[:,0:nvars]
    y_class=data_class.values[:,nvars]
    labels = list(set(y_class.flatten()))
    models = {}
    for label in labels:
        ytmp=y_class.copy()
        ytmp[ytmp!=label]=0
        ytmp[ytmp==label]=1
        kernel = GPy.kern.Matern52(np.shape(x_class)[1], ARD=True)
        m=GPy.models.GPClassification(x_class, ytmp[:, None], kernel=kernel)
        
        m.optimize_restarts(messages=False, robust=True, 
                            num_restarts=2
                            )
        #    else:
        #        m.optimize(messages=True)
        models[label]=m
    print("Classification models built...")
    return models[1]


def run_optimizer_approach_interactive(problem, classification_model, path):
    print("Optimizing...")
    evolver_opt = interactive_optimize(problem, classification_model, gen_per_iter, max_iters, path)
    return evolver_opt.population

def run_optimizer_approach_full(problem, classification_model, selection_type):
    print("Optimizing...")
    evolver_opt = full_optimize(problem, classification_model, gen_per_iter, max_iters, FE_max, selection_type)
    return evolver_opt.population


def evaluate_population_DTLZ(path_to_folder, problem_name, nvars, nobjs, ndim, bound_valL, bound_valU, run):
    
    infile = open(path_to_folder + '/Run_' + str(run), 'rb')
    results_data = pickle.load(infile)
    infile.close()
    individuals = results_data["individuals_solutions"]
    surrogate_objectives = results_data["obj_solutions"]

    N = np.shape(individuals)[0]
    
    boundoldL= 0 # for DTLZ
    boundoldU = 1

    boundvecL = bound_valL* np.ones(nvars)
    boundvecU = bound_valU* np.ones(nvars)
    boundvecL = boundoldL* np.ones(nvars)
    boundvecU = boundoldU* np.ones(nvars)
    boundvecL[:ndim]=np.ones(ndim)*bound_valL
    boundvecU[:ndim]=np.ones(ndim)*bound_valU
    boundL = np.tile(boundvecL,(N,1))
    boundU = np.tile(boundvecU,(N,1))

    testproblem = test_problem_builder(problem_name, n_of_objectives= nobjs, n_of_variables=nvars)
    underlying_objectives = testproblem.evaluate(individuals)[0]

    failed_loc = np.where(np.all(individuals >= boundL, axis=1) & np.all(individuals <= boundU, axis=1))
    stat_success = np.ones((N,1))
    stat_success[failed_loc,0]=0
    underlying_obj_success = underlying_objectives[np.where(stat_success==1)[0],:]
    
    data_class = pd.DataFrame(np.hstack((individuals, stat_success)))
    data_success = pd.DataFrame(np.hstack((individuals[np.where(stat_success==1)[0],:],underlying_obj_success)))
    data_all = pd.DataFrame(np.hstack((np.hstack((individuals, underlying_objectives)),stat_success)))
    data_surrogate = pd.DataFrame(np.hstack((np.hstack((individuals, surrogate_objectives)),stat_success)))
    data_class.to_csv(path_to_folder + '/Run_'+ str(run)+ '_data_class.csv',index=False)
    data_success.to_csv(path_to_folder + '/Run_'+ str(run) + '_data_success.csv',index=False)
    data_all.to_csv(path_to_folder + '/Run_'+ str(run) + '_data_all.csv',index=False)
    data_surrogate.to_csv(path_to_folder + '/Run_'+ str(run) + '_data_surrogate.csv',index=False)

def evaluate_population_DBMOPP(
                                init_folder,
                                problem_spec, 
                                path_to_folder,
                                problem_name,
                                nvars, 
                                nobjs, 
                                n_global_pareto_regions, 
                                constraint_type, 
                                run
                                ):
    problem_folder = init_folder + '/DBMOPP/DBMOPP_problems'
    infile = open(path_to_folder + '/Run_' + str(run), 'rb')
    results_data = pickle.load(infile)
    infile.close()
    infile = open(problem_folder + '/'+ problem_spec, 'rb')
    problem_data = pickle.load(infile)
    infile.close()

    problem = problem_data["simple_problem"]
    testproblem = problem.generate_problem()

    individuals = results_data["individuals_solutions"]
    surrogate_objectives = results_data["obj_solutions"]

    N = np.shape(individuals)[0]
    
    boundoldL= -1 # for DTLZ
    boundoldU = 1

    #boundvecL = bound_valL* np.ones(nvars)
    #boundvecU = bound_valU* np.ones(nvars)
    #boundvecL = boundoldL* np.ones(nvars)
    #boundvecU = boundoldU* np.ones(nvars)
    #boundvecL[:ndim]=np.ones(ndim)*bound_valL
    #boundvecU[:ndim]=np.ones(ndim)*bound_valU
    
    #boundL = np.tile(boundvecL,(N,1))
    #boundU = np.tile(boundvecU,(N,1))

    #testproblem = test_problem_builder(problem_name, n_of_objectives= nobjs, n_of_variables=nvars)

    
    underlying_objectives = testproblem.evaluate(individuals)
    
    failed_loc = np.where(np.any(underlying_objectives[2]<=0, axis=1))
    stat_success = np.ones((N,1))
    stat_success[failed_loc,0]=0
    underlying_objectives = underlying_objectives[0]
    underlying_obj_success = underlying_objectives[np.where(stat_success==1)[0],:]

    #data_failed=underlying_objectives[failed_loc[0],:]
    #obj_vals = obj_val[0]
    #obj_success = underlying_objectives[np.where(stat_success==1)[0],:]
    
    #failed_loc = np.where(np.all(individuals >= boundL, axis=1) & np.all(individuals <= boundU, axis=1))
    #stat_success = np.ones((N,1))
    #stat_success[failed_loc,0]=0
    #underlying_obj_success = underlying_objectives[np.where(stat_success==1)[0],:]
    
    data_class = pd.DataFrame(np.hstack((individuals, stat_success)))
    data_success = pd.DataFrame(np.hstack((individuals[np.where(stat_success==1)[0],:],underlying_obj_success)))
    data_all = pd.DataFrame(np.hstack((np.hstack((individuals, underlying_objectives)),stat_success)))
    data_surrogate = pd.DataFrame(np.hstack((np.hstack((individuals, surrogate_objectives)),stat_success)))
    
    data_class.to_csv(path_to_folder + '/Run_'+ str(run)+ '_data_class.csv',index=False)
    data_success.to_csv(path_to_folder + '/Run_'+ str(run) + '_data_success.csv',index=False)
    data_all.to_csv(path_to_folder + '/Run_'+ str(run) + '_data_all.csv',index=False)
    data_surrogate.to_csv(path_to_folder + '/Run_'+ str(run) + '_data_surrogate.csv',index=False)