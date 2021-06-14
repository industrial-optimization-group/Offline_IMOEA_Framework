from pygmo import fast_non_dominated_sorting as nds
import numpy as np
import pickle
import os
from joblib import Parallel, delayed
from optproblems import dtlz
from non_domx import ndx
#import matlab_wrapper2.matlab_wrapper as matlab_wrapper
import csv
import os

dims = [10]
# dims = 4
############################################
folder_data = 'AM_Samples_109_Final'
problem_testbench = 'DTLZ'
#problem_testbench = 'DDMOPP'

#main_directory = 'Offline_Prob_DDMOPP3'
main_directory = '/home/amrzr/Work/Codes/Tests_Probabilistic_Rev2'
#main_directory = '/home/amrzr/Work/Codes/Tests_CSC_R2_Finalx'
#main_directory = 'Tests_additional_obj1'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'
#main_directory = 'Tests_CSC_4'

#objectives = [4,5,6]
objectives = [2,3,5,8]
#objectives = [2,3,4,5,6,8,10]

#problems = ['DTLZ4']
#problems = ['DTLZ2','DTLZ5','DTLZ6','DTLZ7']
problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2']
#problems = ['P1']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3

#modes = [1, 7, 8]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
#modes = [100,700,800]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [84,74]
#modes = [823,723]
#modes = [12102,72102,82102]
#modes = [1, 7, 8, 12, 72, 82]
#modes = [12, 72, 82]
modes = [800]

#sampling = ['BETA', 'MVNORM']
#sampling = ['LHS']
#sampling = ['BETA','OPTRAND','MVNORM']
#sampling = ['OPTRAND']
#sampling = ['MVNORM']
sampling = ['LHS', 'MVNORM']

#emo_algorithm = ['RVEA','IBEA']
emo_algorithm = ['RVEA']
#emo_algorithm = ['IBEA']
#emo_algorithm = ['NSGAIII']
#emo_algorithm = ['MODEL_CV']




#############################################

nruns = 11
pool_size = 4


def f(name, num_of_objectives_real, num_of_variables, x):
    """The function to predict."""
    if name == "DTLZ1":
        obj_val = dtlz.DTLZ1(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ2":
        obj_val = dtlz.DTLZ2(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ3":
        obj_val = dtlz.DTLZ3(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ4":
        obj_val = dtlz.DTLZ4(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ5":
        obj_val = dtlz.DTLZ5(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ6":
        obj_val = dtlz.DTLZ6(num_of_objectives_real, num_of_variables)(x)

    elif name == "DTLZ7":
        obj_val = dtlz.DTLZ7(num_of_objectives_real, num_of_variables)(x)

    return obj_val


def parallel_execute(run, path_to_file, prob, obj):
    actual_objectives_nds = None
    #print(run)
    path_to_file = path_to_file + '/Run_' + str(run)
    
    try:
        infile = open(path_to_file, 'rb')
        results_data = pickle.load(infile)
        infile.close()
        individual_nds = results_data["individuals_solutions"]
        surrogate_objectives_nds = results_data["obj_solutions"]
        #print(np.shape(individual_nds))
        #print(np.shape(surrogate_objectives_nds))
        path_to_file2 = path_to_file + '_pop'
        with open(path_to_file2, 'w') as f:
            writer = csv.writer(f)
            for line in individual_nds: writer.writerow(line)
        path_to_file3 = path_to_file + '_obj'
        with open(path_to_file3, 'w') as f:
            writer = csv.writer(f)
            for line in surrogate_objectives_nds: writer.writerow(line)
        #print("File written...")
    except Exception as e:
        print(path_to_file + "__" + str(run) + "__" + str(e))
        #try:
        #    os.remove(path_to_file)
        #    print("File deleted")
        #except Exception as e:
        print(str(e))


def parallel_execute_2(run, path_to_file, prob, obj):
    print(run)
    path_to_file1= path_to_file + '/Run_' + str(run)
    infile = open(path_to_file1, 'rb')
    results_data = pickle.load(infile)
    infile.close()
    individual_nds = results_data["individuals_solutions"]
    surrogate_objectives_nds = results_data["obj_solutions"]
    actual_objectives_nds = None
    for i in range(np.shape(individual_nds)[0]):
        if i == 0:
            actual_objectives_nds = np.asarray(f(prob, obj, dims[0], individual_nds[i, :]))
            actual_objectives_nds = actual_objectives_nds.reshape(1,obj)
        else:
            actual_objectives_nds = np.vstack((actual_objectives_nds, f(prob, obj, dims[0], individual_nds[i, :])))
    actual_objectives_nds = np.asarray(actual_objectives_nds)
    print(np.shape(actual_objectives_nds))
    #if np.shape(individual_nds)[0] > 1:
    #    non_dom_front = ndx(actual_objectives_nds)
    #    actual_objectives_nds = actual_objectives_nds[non_dom_front[0][0]]
    #else:
    #    actual_objectives_nds = actual_objectives_nds.reshape(1, obj)
    results_dict = {
        'obj_solns': actual_objectives_nds
    }
    path_to_file2 = path_to_file + '/Run_' + str(run) + '_soln_pickle'
    outfile = open(path_to_file2, 'wb')
    pickle.dump(results_dict, outfile)
    print("File written...")


for samp in sampling:
    for obj in objectives:
        for n_vars in dims:
            for prob in problems:
                for algo in emo_algorithm:
                    for mode in modes:
                        path_to_file = main_directory \
                                       + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                       '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                        #print(path_to_file)
                        if problem_testbench == 'DDMOPP':
                            Parallel(n_jobs=pool_size)(
                                delayed(parallel_execute)(run, path_to_file, prob, obj) for run in range(nruns))
                        else:
                             Parallel(n_jobs=pool_size)(
                                delayed(parallel_execute_2)(run, path_to_file, prob, obj) for run in range(nruns))                           
                        #for run in range(nruns):
                        #    parallel_execute(run, path_to_file, prob, obj)
