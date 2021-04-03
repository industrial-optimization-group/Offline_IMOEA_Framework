import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/Offline_IMOEA_Framework/')
import copy
import numpy as np
import pickle
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import csv
#from IGD_calc import igd, igd_plus
from non_domx import ndx
from pygmo import hypervolume as hv
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import math
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib import rc
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from ranking_approaches import  calc_rank
from desdeo_emo.othertools.EH_metric import Expanding_HC

#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('text', usetex=True)
#plt.rcParams.update({'font.size': 15})


#pareto_front_directory = 'True_Pareto_5000'

data_folder = '/home/amrzr/Work/Codes/data'
#main_directory = 'Offline_Prob_DDMOPP3'
#main_directory = 'Tests_Probabilistic_Finalx_new'
#main_directory = 'Tests_additional_obj1'
#main_directory = 'Tests_Gpy_1'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'
#main_directory = 'Test_Gpy3'
#main_directory = 'Test_DR_4'  #DR = Datatset Reduction
main_directory = 'Test_DR_Scratch'
#main_directory = 'Test_DR_CSC_1'
#main_directory = 'Test_RF'
#main_directory = 'Test_DR_CSC_Final_1'
init_folder = data_folder + '/initial_samples_old'

mod_p_val = True
perform_bonferonni = False
metric = 'HV'
#metric = 'IGD'
save_fig = 'pdf'
#dims = [5,8,10] #,8,10]
#dims = [2, 5, 7, 10]
dims = [10]
#sample_sizes = [2000, 10000]#, 50000]
sample_sizes = [109]

#problem_testbench = 'DTLZ'
problem_testbench = 'DDMOPP'



#objectives = [3,5,7]
objectives = [3]
#objectives = [2,3,5]
#objectives = [3,5,6,8,10]

#problems = ['DTLZ2']
#problems = ['DTLZ2','DTLZ4']
problems = ['P1']
#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2','P3','P4']
#problems = ['P4']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3


sampling = ['LHS']
#sampling = ['MVNORM']
#sampling = ['LHS', 'MVNORM']

emo_algorithm = ['RVEA']


approaches = ["genericRVEA","probRVEA"]
approaches_string = '_'.join(approaches)

approaches_length = int(np.size(approaches))


hv_ref = {"DTLZ2": {"2": [3, 3], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6],  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ4": {"2": [3, 3.1], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6] ,  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ5": {"2": [2.5, 3], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6] ,  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ6": {"2": [10, 10], "3": [20, 20, 20], "5": [20, 20, 20, 20, 20] ,  "7": [20, 20, 20, 20, 20, 20, 20]},
          "DTLZ7": {"2": [1, 20], "3": [1, 1, 40], "5": [1, 1, 1, 1, 50] ,  "7": [1, 1, 1, 1, 1, 1, 70]}}


nruns = 9
pool_size = nruns

plot_boxplot = True

l = [approaches]*nruns
labels = [list(i) for i in zip(*l)]
labels = [y for x in labels for y in x]

p_vals_all_hv = None
p_vals_all_rmse = None
p_vals_all_time = None
index_all = None

for sample_size in sample_sizes:
    for samp in sampling:
        for prob in problems:
            for obj in objectives:
                for n_vars in dims:
                    if (problem_testbench == 'DTLZ' and obj < n_vars) or problem_testbench == 'DDMOPP':
                        
                        fig = plt.figure()
                        fig.set_size_inches(15, 5)
                        #plt.xlim(0, 1)
                        #plt.ylim(0, 1)

                        #if save_fig == 'pdf':
                        #    plt.rcParams["text.usetex"] = True
                        #with open(pareto_front_directory + '/True_5000_' + prob + '_' + obj + '.txt') as csv_file:
                        #    csv_reader = csv.reader(csv_file, delimiter=',')
                        
                        for algo in emo_algorithm:
                            #igd_all = np.zeros([nruns, np.shape(modes)[0]])
                            igd_all = None
                            rmse_mv_all = None
                            solution_ratio_all = None
                            time_taken_all = None
                            path_to_file = data_folder + '/test_runs/'+  main_directory \
                                    + '/Offline_Mode_' + approaches_string + '_' + algo + \
                                    '/' + samp + '/' + str(sample_size) + '/' + problem_testbench  + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                            print(path_to_file)


                            def parallel_execute(run, path_to_file, prob, obj):
                                rmse_mv_iters = {}
                                eh_iters = {}
                                mean_rmse_mv_iters = np.zeros(approaches_length)
                                mean_eh_iters = np.zeros(approaches_length)

                                path_to_file_run = path_to_file + '/Run_' + str(run)
                                path_to_file_evaluated = path_to_file_run + '_evaluated'
                                infile = open(path_to_file_run, 'rb')
                                run_data=pickle.load(infile)
                                infile.close()
                                infile = open(path_to_file_evaluated, 'rb')
                                evaluated_data=pickle.load(infile)
                                infile.close()

                                for iter in range(len(run_data)):
                                    rmse_mv_iters[iter] = {}
                                    eh_iters[iter] = {}
                                    rmse_mv_iters[iter]['phase'] = run_data[iter]['phase']
                                    eh_iters[iter]['phase'] = run_data[iter]['phase']
                                    reference_point_iter = run_data[iter]['reference_point']
                                    set_sizes_all = {}
                                    eh_obj=Expanding_HC()
                                    underlying_objs_temp = None
                                    for approach in approaches:
                                        if underlying_objs_temp is None:
                                            underlying_objs_temp = evaluated_data[iter][approach]
                                        else:
                                            underlying_objs_temp = np.vstack((underlying_objs_temp, evaluated_data[iter][approach]))
                                        
                                    nadir_iter = np.max(underlying_objs_temp, axis=0)
                                    for approach,count in zip(approaches,range(approaches_length)):                
                                        set_sizes_all[count] = eh_obj.hypercube_size(reference_point_iter,evaluated_data[iter][approach],obj,nadir_iter)
                                    eh_approaches_temp = eh_obj.area(set_sizes_all)
                                    mean_eh_iters = np.add(mean_eh_iters, eh_approaches_temp)

                                    for approach,count in zip(approaches,range(approaches_length)): 
                                        eh_iters[iter][approach] = eh_approaches_temp[count]                                        

                                    underlying_objs_temp = None
                                    surrogate_objs_temp = None
                                    rmse_mv_approaches_temp = np.zeros(approaches_length)
                                    for approach,count in zip(approaches,range(approaches_length)): 
                                        underlying_objs_temp = evaluated_data[iter][approach]
                                        surrogate_objs_temp = run_data[iter][approach].objectives
                                        rmse_mv_sols = 0
                                        for i in range(np.shape(surrogate_objs_temp)[0]):
                                            rmse_mv_sols += distance.euclidean(surrogate_objs_temp[i,:], underlying_objs_temp[i,:])
                                        rmse_mv_sols = rmse_mv_sols/np.shape(surrogate_objs_temp)[0]
                                        rmse_mv_iters[iter][approach] = rmse_mv_sols
                                        rmse_mv_approaches_temp[count] = rmse_mv_sols
                                    mean_rmse_mv_iters = np.add(mean_rmse_mv_iters, rmse_mv_approaches_temp)
                                #print("RMSE:",rmse_mv_iters)
                                #print("EH metric:",eh_iters)
                                mean_eh_iters = np.divide(mean_eh_iters,len(run_data))
                                mean_rmse_mv_iters = np.divide(mean_rmse_mv_iters,len(run_data))
                                return [mean_rmse_mv_iters,mean_eh_iters]


                            temp = Parallel(n_jobs=11)(delayed(parallel_execute)(run, path_to_file, prob, obj) for run in range(nruns))
                            temp = np.asarray(temp)
                            #print(temp)
                            mean_rmse = np.transpose(temp[:, 0])
                            mean_eh = np.transpose(temp[:, 1])
                            #print(mean_eh)
                            #print(mean_rmse)



                            """
                            if plot_boxplot is True:
                                if igd_all is None:
                                    igd_all = igd_temp
                                    rmse_mv_all = rmse_mv_sols_temp
                                    solution_ratio_all = solution_ratio_temp
                                    time_taken_all = time_taken_temp
                                else:
                                    igd_all = np.vstack((igd_all, igd_temp))
                                    rmse_mv_all = np.vstack((rmse_mv_all,rmse_mv_sols_temp))                            
                                    solution_ratio_all = np.vstack((solution_ratio_all,solution_ratio_temp))
                                    time_taken_all = np.vstack((time_taken_all,time_taken_temp))

                            igd_all = np.transpose(igd_all)
                            rmse_mv_all = np.transpose(rmse_mv_all)
                            solution_ratio_all = np.transpose(solution_ratio_all)
                            time_taken_all = np.transpose(time_taken_all)

                        
                            lenx = np.zeros(int(math.factorial(mode_length)/((math.factorial(mode_length-2))*2)))
                            p_value_rmse =  copy.deepcopy(lenx)
                            p_value_hv = copy.deepcopy(lenx)
                            p_value_time = copy.deepcopy(lenx)
                            p_cor_temp_hv =  copy.deepcopy(lenx)
                            p_cor_temp_rmse = copy.deepcopy(lenx)
                            p_cor_temp_time = copy.deepcopy(lenx)
                            count = 0
                            count_prev = 0
                            for i in range(mode_length-1):
                                for j in range(i+1,mode_length):
                                    w, p1 = wilcoxon(x=igd_all[:, i], y=igd_all[:, j])
                                    p_value_hv[count] = p1
                                    w, p2 = wilcoxon(x=rmse_mv_all[:, i], y=rmse_mv_all[:, j])
                                    p_value_rmse[count] = p2
                                    w, p3 = wilcoxon(x=time_taken_all[:, i], y=time_taken_all[:, j])
                                    p_value_time[count] = p3
                                    count +=1
                                if perform_bonferonni is True:
                                    if mod_p_val is True:
                                        r, p_cor_temp_hv[count_prev:count], alps, alpb = multipletests(p_value_hv[count_prev:count], alpha=0.05, method='bonferroni',
                                                                            is_sorted=False, returnsorted=False)
                                        r, p_cor_temp_rmse[count_prev:count], alps, alpb = multipletests(p_value_rmse[count_prev:count], alpha=0.05, method='bonferroni',
                                                                            is_sorted=False, returnsorted=False)
                                        r, p_cor_temp_time[count_prev:count], alps, alpb = multipletests(p_value_time[count_prev:count], alpha=0.05, method='bonferroni',
                                                                            is_sorted=False, returnsorted=False)
                                        count_prev = count
                            if perform_bonferonni is True:
                                p_cor_hv = p_cor_temp_hv
                                p_cor_rmse = p_cor_temp_rmse
                                p_cor_time = p_cor_temp_time
                            else:
                                p_cor_hv = p_value_hv
                                p_cor_rmse = p_value_rmse
                                p_cor_time = p_value_time

                            if mod_p_val is False:
                                r, p_cor_hv, alps, alpb = multipletests(p_value_hv, alpha=0.05, method='bonferroni', is_sorted=False,
                                                                    returnsorted=False)
                                r, p_cor_rmse, alps, alpb = multipletests(p_value_rmse, alpha=0.05, method='bonferroni', is_sorted=False,
                                                                    returnsorted=False)
                                r, p_cor_time, alps, alpb = multipletests(p_value_time, alpha=0.05, method='bonferroni', is_sorted=False,
                                                                    returnsorted=False)
                            current_index = [sample_size, samp, prob, obj, n_vars]
                            ranking_hv = calc_rank(p_cor_hv, np.median(igd_all, axis=0),mode_length)
                            ranking_rmse = calc_rank(p_cor_rmse, np.median(rmse_mv_all, axis=0)*-1,mode_length)
                            ranking_time = calc_rank(p_cor_time, np.median(time_taken_all, axis=0)*-1,mode_length)
                            #adding other indicators mean, median, std dev
                            #p_cor = (np.asarray([p_cor, np.mean(igd_all, axis=0),
                            #                             np.median(igd_all, axis=0),
                            #                             np.std(igd_all, axis=0)])).flatten()
                            p_cor_hv = np.hstack((p_cor_hv, np.mean(igd_all, axis=0),
                                                        np.median(igd_all, axis=0),
                                                        np.std(igd_all, axis=0), ranking_hv))
                            p_cor_hv = np.hstack((current_index,p_cor_hv))

                            p_cor_rmse = np.hstack((p_cor_rmse, np.mean(rmse_mv_all, axis=0),
                                                        np.median(rmse_mv_all, axis=0),
                                                        np.std(rmse_mv_all, axis=0), ranking_rmse))
                            p_cor_rmse = np.hstack((current_index,p_cor_rmse))

                            p_cor_time = np.hstack((p_cor_time, np.mean(time_taken_all, axis=0),
                                                        np.median(time_taken_all, axis=0),
                                                        np.std(time_taken_all, axis=0), ranking_time))
                            p_cor_time = np.hstack((current_index,p_cor_time))
                            
                            if p_vals_all_hv is None:
                                p_vals_all_hv = p_cor_hv
                                p_vals_all_rmse = p_cor_rmse
                                p_vals_all_time = p_cor_time
                            else:
                                p_vals_all_hv = np.vstack((p_vals_all_hv, p_cor_hv))
                                p_vals_all_rmse = np.vstack((p_vals_all_rmse, p_cor_rmse))
                                p_vals_all_time = np.vstack((p_vals_all_time, p_cor_time))
                            


                            ax = fig.add_subplot(131)
                            bp = ax.boxplot(igd_all, showfliers=False, widths=0.45)
                            #ax.set_title(metric + '_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars))
                            ax.set_title('Hypervolume comparison')
                            ax.set_xlabel('Approaches')
                            #ax.set_ylabel(metric)
                            ax.set_ylabel('Hypervolume')
                            ax.set_xticklabels(approaches, rotation=75, fontsize=10)

                            ax = fig.add_subplot(132)
                            bp = ax.boxplot(rmse_mv_all, showfliers=False, widths=0.45)
                            #ax.set_title(metric + '_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars))
                            ax.set_title('Accuracy comparison')
                            ax.set_xlabel('Approaches')
                            #ax.set_ylabel(metric)
                            ax.set_ylabel('RMSE_MV')
                            ax.set_xticklabels(approaches, rotation=75, fontsize=10)

                            print(time_taken_all)
                            ax = fig.add_subplot(133)
                            bp = ax.boxplot(time_taken_all, showfliers=False, widths=0.45)
                            #ax.set_title(metric + '_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars))
                            ax.set_title('Building time comparison')
                            ax.set_xlabel('Approaches')
                            #ax.set_ylabel(metric)
                            ax.set_ylabel('Time(s)')
                            ax.set_xticklabels(approaches, rotation=75, fontsize=10)

                            if problem_testbench is 'DTLZ':
                                filename_fig = './data/test_runs/' + main_directory + '/' + metric + '_' + str(sample_size) + '_' + samp + '_' + algo + '_' + prob + '_' + str(
                                    obj) + '_' + str(n_vars)
                            else:
                                filename_fig = './data/test_runs/' + main_directory + '/' + metric + '_' + str(sample_size) + '_' + samp + '_' + algo + '_' + prob + '_' + str(
                                    obj) + '_' + str(n_vars)

                            if save_fig == 'png':
                                fig.savefig(filename_fig + '.png', bbox_inches='tight')
                            else:
                                fig.savefig(filename_fig + '.pdf', bbox_inches='tight')
                            ax.clear()


if mod_p_val is False:
    file_summary = './data/test_runs/' + main_directory + '/Summary_HV_' + problem_testbench + '.csv'
else:
    file_summary = './data/test_runs/' + main_directory + '/Summary_HV_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_hv)
writeFile.close()

if mod_p_val is False:
    file_summary = './data/test_runs/' + main_directory + '/Summary_RMSE_' + problem_testbench + '.csv'
else:
    file_summary = './data/test_runs/' + main_directory + '/Summary_RMSE_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_rmse)
writeFile.close()

if mod_p_val is False:
    file_summary = './data/test_runs/' + main_directory + '/Summary_TIME_' + problem_testbench + '.csv'
else:
    file_summary = './data/test_runs/' + main_directory + '/Summary_TIME_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_time)
writeFile.close()
"""


