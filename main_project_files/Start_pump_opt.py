import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/Offline_IMOEA_Framework/')

from desdeo_problem.Problem import DataProblem

from desdeo_problem.testproblems.TestProblems import test_problem_builder
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from main_project_files.surrogate_fullGP import FullGPRegressor as fgp

#from pygmo import non_dominated_front_2d as nd2
from non_domx import ndx
import scipy.io
from sklearn.neighbors import NearestNeighbors
import time
import GPy
from BIOMA_framework import interactive_optimize

max_iters = 5
gen_per_iter=10
nobjs = 3 
nvars = 22
main_directory = 'Pump_Test_1'
data_folder = '/home/amrzr/Work/Codes/data'
data_file = data_folder+'/pump_data/01_DOE_data.csv'
path = data_folder + '/test_runs/' + main_directory

df = pd.read_csv(data_file)
df[['f1','f2','f3']] = df[['f1','f2','f3']]*-1

def run_optimizer_approach_interactive(problem, path):
    print("Optimizing...")
    evolver_opt = interactive_optimize(problem, gen_per_iter, max_iters, path)
    return evolver_opt.population

def build_surrogates(nobjs, nvars, df): #x_data, y_data):
    x_names = list(df.columns)[0:22]
    y_names = list(df.columns)[22:25]
    row_names = ['lower_bound','upper_bound']
    #data = pd.DataFrame(np.hstack((x_data,y_data)), columns=x_names+y_names)
    data = df
    print(data)
    x_low = [20, 0.2, 0.22, 0.25, -5, 85, 355, 450, 15, 15, -10, 16, 0.25, 0.2, 0.25, -5, 85, 450, 15, 15, 27, -15]
    x_high = [30,0.72, 0.76 , 0.8 , 0 , 90, 380 , 600, 45 , 50,10 , 26, 0.76, 0.7 ,0.76, 0 ,90 ,600 ,60 ,50 ,35 ,5]
    bounds = pd.DataFrame(np.vstack((x_low,x_high)), columns=x_names, index=row_names)
    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names,bounds=bounds)
    start = time.time()
    problem.train(fgp)
    end = time.time()
    time_taken = end - start
    return problem




surrogate_problem = build_surrogates(nobjs, nvars, df)
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






