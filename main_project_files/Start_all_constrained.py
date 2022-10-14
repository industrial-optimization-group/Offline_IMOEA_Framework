from mimetypes import init
import main_execute_constrained_tests as mexeConst
import pickle
#import pickle_to_mat_converter as pickmat
#from AMD_data_evaluate import evaluate_population
import os
from joblib import Parallel, delayed
import datetime
import traceback
import pandas as pd
import numpy as np

data_folder = '/home/amrzr/Work/Codes/data'
results_folder = 'Test_constrained_4'
init_folder = data_folder + '/constraint_handling_dataset'

file_exists_check = True
#file_exists_check = False

evaluate_data = True
#evaluate_data = False


#problem_testbench = 'DTLZ'
problem_testbench = 'DBMOPP'
#problem_testbench = 'GAA'

#file_instances = init_folder + '/test_instances2.csv'
file_instances = init_folder + '/test_instances_DBMOPP.csv'

data_instances = pd.read_csv(file_instances)
all_problems = data_instances["problem"].values
all_n_vars = data_instances["nvars"].values
all_objs = data_instances["K"].values
all_sample_size = data_instances["N"].values
if problem_testbench == 'DTLZ':
    all_ndim = data_instances["ndim"].values
    all_bound_valL = data_instances["bound_valL"].values
    all_bound_valU = data_instances["bound_valU"].values
else:
    all_n_global_pareto_regions = data_instances["n_global_pareto_regions"].values
    all_constraint_type = data_instances["constraint_type"].values

"""
all_problems = ['DTLZ2']
#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1']

all_n_vars = [5]

all_objs = [2]

#all_sampling = ['LHS']

all_sample_size = [250]

all_ndim = [5]

all_bound_valL = [0.2]

all_bound_valU = [0.9]
"""

approaches = ["genericRVEA","probRVEA","probRVEA_constraint_v1","probRVEA_constraint_v2"]

size_instance = np.size(all_problems)

interactive = False

#############################################


nruns = 11
parallel_jobs = 3
log_time = str(datetime.datetime.now())


def parallel_execute(run, instance, approach):
    
    prob = all_problems[instance]
    n_vars = all_n_vars[instance]
    obj = all_objs[instance]
    #samp = all_sampling[instance]
    sample_size = all_sample_size[instance]
    if problem_testbench == 'DTLZ':
        ndim = all_ndim[instance]
        bound_valL = all_bound_valL[instance]
        bound_valU = all_bound_valU[instance]
        problem_spec = prob +'_'+ str(sample_size) + '_' + str(obj) + '_' + \
                    str(n_vars) +  '_b'+str(ndim) +'_' + str(bound_valL).replace('.','') + \
                        str(bound_valU).replace('.','')
    else:
        n_global_pareto_regions = all_n_global_pareto_regions[instance]
        constraint_type = all_constraint_type[instance]
        problem_spec = prob +'_'+ str(sample_size) + '_' + str(obj) + '_' + \
        str(n_vars) +  '_b'+str(n_global_pareto_regions) +'_' + str(constraint_type)
    

    path_to_file = data_folder + '/test_runs/'+  results_folder \
                + '/' + approach + '/' + problem_spec

    print(path_to_file)
    
    with open(data_folder + '/test_runs/'+ results_folder +"/log_"+log_time+".txt", "a") as text_file:
        text_file.write("\n"+path_to_file+"___"+str(run)+"___Started___"+str(datetime.datetime.now()))
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)
        print("Creating Directory...")

    if evaluate_data is False:
        if (problem_testbench == 'DTLZ' and obj < n_vars) or problem_testbench == 'DBMOPP':
            path_to_file = path_to_file + '/Run_' + str(run)
            if os.path.exists(path_to_file) is False or file_exists_check is False:
                print('Starting Run!')
                try:
                    results_dict = mexeConst.run_optimizer(problem_testbench=problem_testbench,
                                                        init_folder = init_folder,
                                                        nvars = n_vars, 
                                                        nobjs = obj,
                                                        problem_spec=problem_spec,
                                                        approach=approach,
                                                        run=run)
                    outfile = open(path_to_file, 'wb')
                    pickle.dump(results_dict, outfile)
                    outfile.close()
                    with open(data_folder + '/test_runs/'+results_folder+"/log_"+log_time+".txt", "a") as text_file:
                            text_file.write("\n"+path_to_file+"___"+str(run)+"___Ended___"+str(datetime.datetime.now()))
                except Exception as e:
                    print(e)
                    with open(data_folder + '/test_runs/'+results_folder+"/log_"+log_time+".txt", "a") as text_file:
                        text_file.write("\n"+path_to_file+"___"+str(run)+ "-Error-"+str(e) + "______" 
                        + traceback.format_exc()+ "________" + str(datetime.datetime.now()))   
                                
            else:
                with open(data_folder + '/test_runs/'+results_folder+"/log_"+log_time+".txt", "a") as text_file:
                    text_file.write("\n"+path_to_file+"___"+str(run)+"___File already exists!___"+str(datetime.datetime.now()))
                    print('File already exists!')
    else:
        try:
            if problem_testbench == 'DTLZ':
                mexeConst.evaluate_population_DTLZ(path_to_folder = path_to_file,
                                                problem_name = prob,
                                                nvars = n_vars,
                                                nobjs = obj,
                                                ndim = ndim,
                                                bound_valL = bound_valL,
                                                bound_valU = bound_valU,
                                                run=run)
            else:
                mexeConst.evaluate_population_DBMOPP(init_folder = init_folder,
                                                problem_spec = problem_spec,
                                                path_to_folder = path_to_file,
                                                problem_name = prob,
                                                nvars = n_vars,
                                                nobjs = obj,
                                                n_global_pareto_regions = n_global_pareto_regions,
                                                constraint_type = constraint_type,
                                                run=run)                
            
        except Exception as e:
            print(e)

    with open(data_folder + '/test_runs/'+results_folder+"/log_"+log_time+".txt", "a") as text_file:
        text_file.write("\n"+path_to_file+"___"+str(run)+"___Ended___"+str(datetime.datetime.now()))


try:
    temp = Parallel(n_jobs=parallel_jobs)(
        delayed(parallel_execute)(run, instance, approach)        
        for run in range(nruns)
        for approach in approaches
        for instance in range(size_instance))
#    for run in range(nruns):
#        parallel_execute(run, path_to_file)
except Exception as e:
    print(e)
    with open(data_folder + '/test_runs/'+results_folder+"/log_"+log_time+".txt", "a") as text_file:
        text_file.write("\n"+ str(e) + "______" + str(datetime.datetime.now()))   


