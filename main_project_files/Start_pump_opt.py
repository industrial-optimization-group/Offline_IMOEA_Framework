import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/Offline_IMOEA_Framework/')

from desdeo_problem.Problem import DataProblem

from desdeo_problem.testproblems.TestProblems import test_problem_builder
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main_project_files.surrogate_fullGP import FullGPRegressor as fgp
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

#from pygmo import non_dominated_front_2d as nd2
from non_domx import ndx
import scipy.io
from sklearn.neighbors import NearestNeighbors
import time
import GPy
#from BIOMA_framework import interactive_optimize
from BIOMA_framework_worst import interactive_optimize
import copy

max_iters = 5
gen_per_iter=10
nobjs = 3 
nvars = 22
#main_directory = 'Pump_Test_Tomas_2_140'
main_directory = 'Pump_Test_Tomas_3_140_all'
data_folder = '/home/amrzr/Work/Codes/data'
#data_file = data_folder+'/pump_data/01_DOE_data.csv'
#data_file = data_folder+'/pump_data/02_DOE_140_data.csv'
data_file = data_folder+'/pump_data/03_DOE_140_all_data.csv'
path = data_folder + '/test_runs/' + main_directory

df = pd.read_csv(data_file)
df[['f1','f2','f3']] = df[['f1','f2','f3']]*-1

#x_low = [20, 0.2, 0.22, 0.25, -5, 85, 355, 450, 15, 15, -10, 16, 0.25, 0.2, 0.25, -5, 85, 450, 15, 15, 27, -15]
#x_high = [30, 0.72, 0.76 , 0.8 , 0 , 90, 380 , 600, 45 , 50,10 , 26, 0.76, 0.7 ,0.76, 0 ,90 ,600 ,60 ,50 ,35 ,5]

x_low = np.ones(22)*0
x_high = np.ones(22)

x_low_new = np.ones(22)*0
x_high_new = np.ones(22)


def run_optimizer_approach_interactive(problem, path):
    print("Optimizing...")
    evolver_opt = interactive_optimize(problem, gen_per_iter, max_iters, path)
    return evolver_opt.population

def scale_data(data):
    x_data = np.asarray(data.loc[:,:'x22'])
    x_data_scaled = (x_data - x_low)/(np.asarray(x_high) - np.asarray(x_low))
    data.loc[:,:'x22'] = x_data_scaled
    return data

def build_surrogates(nobjs, nvars, df): #x_data, y_data):
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

surrogate_problem = build_surrogates(nobjs, nvars, data_scaled)
#print(surrogate_problem.objectives[2]._model.predict(np.asarray(data_scaled.loc[0:1,:'x22'])))
#print(surrogate_problem.objectives[2]._model.predict(np.asarray(x_low_new).reshape(1,-1)))
#print(surrogate_problem.objectives[2]._model.predict(zz))
population = run_optimizer_approach_interactive(surrogate_problem, path)

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






