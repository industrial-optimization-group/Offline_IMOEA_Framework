# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 18:20:05 2022

@author: Jana
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/Offline_IMOEA_Framework/')
#sys.path.insert(1, 'C:\\Users\\Jana\\Pump\\Offline4\\Offline_IMOEA_Framework-main\\')
# from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_problem.testproblems.DBMOPP.DBMOPP_generator import DBMOPP_generator
from scipy.stats import qmc

import copy
import pickle
import os

# problem parameters
problem_name = 'DBMOPP'
nobjs = 2
nvars = 2
N = 250
bound_valL=0.2
bound_valU=0.9

boundoldL= -1 # for DBMOPP
boundoldU = 1  
ndim = nvars # nvars...for box constraints, 1,2...for dimensionwise constraints

n_local_pareto_regions = 0
n_disconnected_regions = 0 
n_global_pareto_regions =1
const_space = 0.0 
pareto_set_type = 0
constraint_type = 0 
ndo = 0
vary_sol_density = False
vary_objective_scales = False
prop_neutral = 0
nm = 10000
        


# end of parameters


cwd=os.getcwd()
path = cwd+'\\Jana_tests_DBMOPP\\DataSets\\'

problem_spec ='tests_' + problem_name +'_'+ str(N) + '_' + str(nobjs) + '_' + \
    str(nvars) +  '_b'+str(ndim) +'_' + str(bound_valL).replace('.','') + \
        str(bound_valU).replace('.','')
 
boundvecL = bound_valL* np.ones(nvars)
boundvecU = bound_valU* np.ones(nvars)
boundvecL = boundoldL* np.ones(nvars)
boundvecU = boundoldU* np.ones(nvars)
boundvecL[:ndim]=np.ones(ndim)*bound_valL
boundvecU[:ndim]=np.ones(ndim)*bound_valU
boundL = np.tile(boundvecL,(N,1))
boundU = np.tile(boundvecU,(N,1))

sampler = qmc.LatinHypercube(d=nvars)
data = sampler.random(n=N)*(boundoldU-(boundoldL)) + (boundoldL)
# print(data)



simple_problem = DBMOPP_generator(nobjs,nvars,
        n_local_pareto_regions,
        n_disconnected_regions,
        n_global_pareto_regions,
        const_space,
        pareto_set_type,
        constraint_type, 
        ndo, vary_sol_density, vary_objective_scales, prop_neutral, nm)
testproblem=simple_problem.generate_problem()

obj_val = testproblem.evaluate(data)


failed_loc = np.where(np.all(data >= boundL, axis=1) & np.all(data <= boundU, axis=1))
stat_success = np.ones((N,1))
stat_success[failed_loc,0]=0

data_failed=data[failed_loc[0],:]
#ax.scatter(data_failed[:,0],data_failed[:,1],data_failed[:,2], c='black', marker=',')
obj_vals = obj_val[0]
obj_success = obj_vals[np.where(stat_success==1)[0],:]

np.size(failed_loc)
print(np.size(failed_loc))


data_class = pd.DataFrame(np.hstack((data, stat_success)))
data_success = pd.DataFrame(np.hstack((data[np.where(stat_success==1)[0],:],obj_success)))
data_all = pd.DataFrame(np.hstack((np.hstack((data, obj_vals)),stat_success)))

data_class.to_csv(path + problem_spec + '_data_class.csv',index=False)
data_success.to_csv(path + problem_spec + '_data_success.csv',index=False)
data_all.to_csv(path + problem_spec + '_data_all.csv',index=False)

plt.rcParams["figure.figsize"] = (10,8)
simple_problem.plot_problem_instance()
plt.show()

# testproblem_copy = copy.deepcopy(testproblem)
#     #classification_model_copy = copy.deepcopy(classification_model)
# models_dict = {
#     'problem_nvars': testproblem_copy.n_of_variables,
#     'problem_nobjs': testproblem_copy.n_of_objectives,
#     'problem_name': problem_name
#             }

# # path = data_folder + '/test_runs/Pump_models'
# path = cwd+'\\Jana_tests_DBMOPP\\problem_models'
# model_file=problem_spec
# outfile = open(path+'/' + model_file, 'wb')
# pickle.dump(models_dict, outfile)
# outfile.close()