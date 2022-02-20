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

max_iters = 10 #50
gen_per_iter= 25 #10
nobjs = 3 
nvars = 22
is_interact = False

#main_directory = 'Pump_Test_Tomas_2_140'
#main_directory = 'Pump_test_140_prob_1'  #180 samples
#main_directory = 'Pump_test_All_probclass_1'   # all samples including test runs
#main_directory = 'Pump_test_All_probclass_2'   # all samples including test runs (new constraint handling /changed transpose in prob class)
#main_directory = 'Pump_test_All_probclass_3'   # all samples including test runs (changed transpose in prob class)
#main_directory = 'Pump_test_01_03_04_prob_1' # all DOE data
#main_directory = 'Pump_test_04_DOE_prob_only_run1'  # 487 samples
#main_directory = 'Pump_test_04_DOE_probclass_v1_run1'  # 487 samples
main_directory = 'Pump_test_04_DOE_probclass_v2_run2'  # 487 samples
#main_directory = 'Pump_test_04_DOE_generic_run1'  # 487 samples
#main_directory = 'Pump_test_140_probclass_2'

data_folder = '/home/amrzr/Work/Codes/data'
#data_file = data_folder+'/pump_data/01_DOE_data.csv'
#data_file = data_folder+'/pump_data/02_DOE_140_data.csv'
data_file = data_folder+'/pump_data/04_DOE_successful.csv'
#data_file = data_folder+'/pump_data/DOE_01_03_04_successful.csv'
#data_file = data_folder+'/pump_data/DataSets_successful.csv'
#data_file = data_folder+'/pump_data/03_DOE_140_all_data.csv'
path = data_folder + '/test_runs/' + main_directory

#selection_type = 'prob_only'  # prob using MC samples (no classification)
#selection_type = 'prob_class_v1' # prob classification using MC samples (product of probabilities)
selection_type = 'prob_class_v2' # prob classification using MC samples (RVEA type constraint handling)
#selection_type = 'generic'

model_file = 'Dataset_04_DOE'
read_saved_models = True

df = pd.read_csv(data_file)
df[['f1','f2','f3']] = df[['f1','f2','f3']]*-1

#x_low = [20, 0.2, 0.22, 0.25, -5, 85, 355, 450, 15, 15, -10, 16, 0.25, 0.2, 0.25, -5, 85, 450, 15, 15, 27, -15]
#x_high = [30, 0.72, 0.76 , 0.8 , 0 , 90, 380 , 600, 45 , 50,10 , 26, 0.76, 0.7 ,0.76, 0 ,90 ,600 ,60 ,50 ,35 ,5]

x_low = np.ones(22)*0
x_high = np.ones(22)

x_low_new = np.ones(22)*0
x_high_new = np.ones(22)

def build_classification_failed():
    main_directory = 'Pump_Test_Tomas_6_177'
    data_folder = '/home/amrzr/Work/Codes/data'
    #data_file = data_folder+'/pump_data/sim_stat2.csv'
    #data_file = data_folder+'/pump_data/03_DOE_180_failed.csv'
    data_file = data_folder+'/pump_data/04_DOE_table_all.csv'
    #data_file = data_folder+'/pump_data/DOE_01_03_04_all.csv'
    #data_file = data_folder+'/pump_data/DataSets_all.csv'
    df = pd.read_csv(data_file)   
    X=df.values[:,0:22]
    y=df.values[:,22]

    labels = list(set(y.flatten()))
    models = {}
    for label in labels:
        ytmp=y.copy()
        ytmp[ytmp!=label]=0
        ytmp[ytmp==label]=1
        kernel = GPy.kern.Matern52(np.shape(X)[1], ARD=True)
        m=GPy.models.GPClassification(X, ytmp[:, None], kernel=kernel)
        
        m.optimize_restarts(messages=False, robust=True, 
                            num_restarts=4
                            )
        #    else:
        #        m.optimize(messages=True)
        models[label]=m
    return models[1]


def run_optimizer_approach_interactive(problem, classification_model, path):
    print("Optimizing...")
    evolver_opt = interactive_optimize(problem, classification_model, gen_per_iter, max_iters, path)
    return evolver_opt.population

def run_optimizer_approach_full(problem, classification_model, path, selection_type):
    print("Optimizing...")
    evolver_opt = full_optimize(problem, classification_model, gen_per_iter, max_iters, path, selection_type)
    return evolver_opt.population

def scale_data(data):
    x_data = np.asarray(data.loc[:,:'x22'])
    x_data_scaled = (x_data - x_low)/(np.asarray(x_high) - np.asarray(x_low))
    data.loc[:,:'x22'] = x_data_scaled
    return data

def build_surrogates(nobjs, nvars, df): #x_data, y_data):
    print("Building surrogates...")
    x_names = list(df.columns)[0:22]
    y_names = list(df.columns)[22:25]
    row_names = ['lower_bound','upper_bound']
    #data = pd.DataFrame(np.hstack((x_data,y_data)), columns=x_names+y_names)
    data = df
    print(data)
    zz = np.asarray(df.loc[:,:'x22'])
    #x_low = np.min(zz,axis=0)
    #x_high = np.max(zz,axis=0)
    #x_low = [20, 0.2, 0.22, 0.25, -5, 85, 355, 450, 15, 15, -10, 16, 0.25, 0.2, 0.25, -5, 85, 450, 15, 15, 27, -15]
    #x_high = [30,0.72, 0.76 , 0.8 , 0 , 90, 380 , 600, 45 , 50,10 , 26, 0.76, 0.7 ,0.76, 0 ,90 ,600 ,60 ,50 ,35 ,5]
    

    bounds = pd.DataFrame(np.vstack((x_low_new,x_high_new)), columns=x_names, index=row_names)
    problem = DataProblem(data=df, variable_names=x_names, objective_names=y_names,bounds=bounds)
    start = time.time()
    problem.train(fgp)
    #problem.train(GaussianProcessRegressor) #, {"kernel": Matern(nu=3/2)})
    end = time.time()
    time_taken = end - start
    return problem

data_scaled = scale_data(df)
#zz = copy.deepcopy(np.asarray(data_scaled.loc[0:1,:'x22']))
#zz[0,0]=zz[0,0]+0.01
#x_low = np.min(zz,axis=0)


if read_saved_models:
    data_file = data_folder+ '/test_runs/Pump_models/' + model_file
    infile = open(data_file, 'rb')
    results_data = pickle.load(infile)
    infile.close()
    surrogate_problem = results_data["surrogate_problem"]
    print("loaded saved models...")
else:
    surrogate_problem = build_surrogates(nobjs, nvars, data_scaled)
classification_model = build_classification_failed()
#print(surrogate_problem.objectives[2]._model.predict(np.asarray(data_scaled.loc[0:1,:'x22'])))
#print(surrogate_problem.objectives[2]._model.predict(np.asarray(x_low_new).reshape(1,-1)))
#print(surrogate_problem.objectives[2]._model.predict(zz))
if is_interact:
    population = run_optimizer_approach_interactive(surrogate_problem, classification_model, path)
else:
    population = run_optimizer_approach_full(surrogate_problem, classification_model, path, selection_type)

results_dict = {
        'individual_archive': population.individuals_archive,
        'objectives_archive': population.objectives_archive,
        'uncertainty_archive': population.uncertainty_archive,
        'individuals_solutions': population.individuals,
        'obj_solutions': population.objectives,
        'uncertainty_solutions': population.uncertainity                
    }
outfile = open(path+'/run_pump', 'wb')
pickle.dump(results_dict, outfile)
outfile.close()


if read_saved_models is False:
    surrogate_problem_copy = copy.deepcopy(surrogate_problem)
    #classification_model_copy = copy.deepcopy(classification_model)
    models_dict = {
            'surrogate_problem': surrogate_problem_copy
            #'classification_model': classification_model_copy           
        }

    path = data_folder + '/test_runs/Pump_models'
    outfile = open(path+'/' + model_file, 'wb')
    pickle.dump(models_dict, outfile)
    outfile.close()






