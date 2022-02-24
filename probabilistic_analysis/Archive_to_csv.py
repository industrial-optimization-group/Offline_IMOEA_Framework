from pygmo import fast_non_dominated_sorting as nds
import numpy as np
import pickle
import os
from joblib import Parallel, delayed
from non_domx import ndx
from optproblems import dtlz
#import matlab_wrapper2.matlab_wrapper as matlab_wrapper
import csv

dims = [10]
# dims = 4
############################################
folder_data = 'AM_Samples_109_Final'

problem_testbench = 'DTLZ'
#problem_testbench = 'DDMOPP'

#main_directory = 'Tests_soln_plot'
#main_directory = 'Offline_Prob_DDMOPP3'
main_directory = '/home/amrzr/Work/Codes/Tests_Probabilistic_Rev2'
#main_directory = 'Tests_CSC_3'
#main_directory = 'Tests_additional_obj1'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'

#objectives = [2]
objectives = [2,3,5,8]
#objectives = [2,3,4,5,6,8,10]

#problems = ['DTLZ7']
problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2']
#problems = ['P1']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3

#modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
#modes = [1,7,8]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [1]
modes = [1,7,8,12,72,82]

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

nruns = 31
pool_size = 4
del_file = False

def fx(name, num_of_objectives_real, num_of_variables, x):
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


def parallel_execute(run, path_to_file, prob, obj,mode):
    actual_objectives = None
    print(run)
    path_to_file = path_to_file + '/Run_' + str(run)
    if del_file is False:
        infile = open(path_to_file, 'rb')
        results_data = pickle.load(infile)
        infile.close()
        individual_archive = results_data["individual_archive"]
        objective_archive = results_data["objectives_archive"]
        #print(objective_archive)
        uncertainty_archive = results_data["uncertainty_archive"]
        if type(individual_archive) is dict:
            individual_archive=np.vstack(individual_archive.values())
            objective_archive=np.vstack(objective_archive.values())
            uncertainty_archive=np.vstack(uncertainty_archive.values())
        print(np.shape(individual_archive))
        print(np.shape(objective_archive))
        print("......")

        if problem_testbench == 'DTLZ':
            for i in range(np.shape(individual_archive)[0]):
                if i == 0:
                    actual_objectives = np.asarray(fx(prob, obj, dims[0], individual_archive[i, :]))
                else:
                    actual_objectives = np.vstack((actual_objectives, fx(prob, obj, dims[0], individual_archive[i, :])))
            path_to_file4 = path_to_file + '_soln_all'
            with open(path_to_file4, 'w') as f:
                writer = csv.writer(f)
                for line in actual_objectives: writer.writerow(line)
        else:
            path_to_file2 = path_to_file + '_popx_all'
            with open(path_to_file2, 'w') as f:
                writer = csv.writer(f)
                for line in individual_archive: writer.writerow(line)

            path_to_file3 = path_to_file + '_surrx_all'
            with open(path_to_file3, 'w') as f:
                writer = csv.writer(f)
                for line in objective_archive: writer.writerow(line)
            
            path_to_file3 = path_to_file + '_uncx_all'
            with open(path_to_file3, 'w') as f:
                writer = csv.writer(f)
                for line in uncertainty_archive: writer.writerow(line)
        
        print("File written...")
    else:
        path_to_file1 = path_to_file + '_solnx_all'
        path_to_file2 = path_to_file + '_popx_all'
        path_to_file3 = path_to_file + '_surrx_all'
        path_to_file4 = path_to_file + '_uncx_all'
        try:
            os.remove(path_to_file1)
        except Exception as e:
            print(str(e))
        try:
            os.remove(path_to_file2)
        except Exception as e:
            print(str(e))
        try:
            os.remove(path_to_file3)
        except Exception as e:
            print(str(e))
        try:
            os.remove(path_to_file4)
        except Exception as e:
            print(str(e))
        print("Files deleted")


    



for samp in sampling:
    for obj in objectives:
        for n_vars in dims:
            for prob in problems:
                for algo in emo_algorithm:
                    for mode in modes:
                        path_to_file = main_directory \
                                       + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                       '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                        print(path_to_file)

                        Parallel(n_jobs=pool_size)(
                            delayed(parallel_execute)(run, path_to_file, prob, obj,mode) for run in range(nruns))

                        #for run in range(nruns):
                        #    parallel_execute(run, path_to_file, prob, obj,mode)
