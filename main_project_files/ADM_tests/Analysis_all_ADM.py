import sys

from scipy.linalg.misc import norm
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
from desdeo_emo.othertools import R_metric as rm
from sklearn.preprocessing import Normalizer
import traceback

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
#main_directory = 'Test_DR_Scratch'
#main_directory = 'Test_Interactive_2'
main_directory = 'Test_Interactive_new1'
#main_directory = 'Test_DR_CSC_1'
#main_directory = 'Test_RF'
#main_directory = 'Test_DR_CSC_Final_1'
init_folder = data_folder + '/initial_samples_109'

mod_p_val = True
perform_bonferonni = True
metric = 'HV'
#metric = 'IGD'
save_fig = 'pdf'
#dims = [5,8,10] #,8,10]
#dims = [2, 5, 7, 10]
dims = [10]
#sample_sizes = [2000, 10000]#, 50000]
sample_sizes = [109]

objectives = [5,7,9]
#objectives = [5]
#objectives = [2,3,5]
#objectives = [3,5,6,8,10]

#problem_testbench = 'DTLZ'
problem_testbench = 'DDMOPP'

#problems = ['DTLZ5']
#problems = ['DTLZ2','DTLZ4']
#problems = ['P1']
#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
problems = ['P1','P2','P3','P4','P5']
#problems = ['P2','P3','P4','P5']
#problems = ['P1','P2']
#problems = ['P4']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3


#sampling = ['LHS']
#sampling = ['MVNORM']
sampling = ['LHS', 'MVNORM']

emo_algorithm = ['RVEA']


#approaches = ["genericRVEA","probRVEA"]
approaches = ["genericRVEA_0","probRVEA_0","genericRVEA_1","probRVEA_1"]
approaches_labels = ["genericRVEA0","probRVEA0","genericRVEA1","probRVEA1"]
approaches_string = '_'.join(approaches)
mode_length = int(np.size(approaches))

approaches_length = int(np.size(approaches))


hv_ref = {"DTLZ2": {"2": [3, 3], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6],  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ4": {"2": [3, 3.1], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6] ,  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ5": {"2": [2.5, 3], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6] ,  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ6": {"2": [10, 10], "3": [20, 20, 20], "5": [20, 20, 20, 20, 20] ,  "7": [20, 20, 20, 20, 20, 20, 20]},
          "DTLZ7": {"2": [1, 20], "3": [1, 1, 40], "5": [1, 1, 1, 1, 50] ,  "7": [1, 1, 1, 1, 1, 1, 70]}}


nruns = 11
pool_size = 2

plot_boxplot = True

l = [approaches]*nruns
labels = [list(i) for i in zip(*l)]
labels = [y for x in labels for y in x]

p_vals_all_eh = None
p_vals_all_rmse = None
p_vals_all_eh_sum = None
p_vals_all_rmse_sum = None
p_vals_all_rhv = None
p_vals_all_rhv_sum = None
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
                                eh_iters_surrogate = {}
                                rhv_iters = {}
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
                                    eh_iters_surrogate[iter] = {}
                                    rhv_iters[iter] = {}
                                    rmse_mv_iters[iter]['phase'] = run_data[iter]['phase']
                                    eh_iters[iter]['phase'] = run_data[iter]['phase']
                                    eh_iters_surrogate[iter]['phase'] = run_data[iter]['phase']
                                    rhv_iters[iter]['phase'] = run_data[iter]['phase']
                                    reference_point_iter = run_data[iter]['reference_point']
                                    set_sizes_all = {}
                                    set_sizes_all_surrogate = {}
                                    eh_obj=Expanding_HC()
                                    eh_obj_surrogate = Expanding_HC()

                                    ref_point = reference_point_iter.reshape(1, obj)
                                    rp_transformer = Normalizer().fit(ref_point)
                                    norm_rp = rp_transformer.transform(ref_point)
                                    rmetric = rm.RMetric(norm_rp, delta=0.2)

                                    #rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                                    #norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

                                    #nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
                                    #norm_nsga = nsga_transformer.transform(int_nsga.population.objectives)

                                    #rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_nsga)
                                    #rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_rvea)


                                    underlying_objs_temp = None
                                    surrogate_objs_temp = None
                                    underlying_objs_norm_other = None
                                    underlying_objs_norm_approach = None


                                    for approach in approaches:
                                        if underlying_objs_temp is None:
                                            underlying_objs_temp = evaluated_data[iter][approach]
                                        else:
                                            underlying_objs_temp = np.vstack((underlying_objs_temp, evaluated_data[iter][approach]))
                                    
                                    for approach in approaches:
                                        if surrogate_objs_temp is None:
                                            surrogate_objs_temp = run_data[iter][approach].objectives
                                        else:
                                            surrogate_objs_temp = np.vstack((surrogate_objs_temp, run_data[iter][approach].objectives))
                                    
                                    nadir_iter = np.max(underlying_objs_temp, axis=0)
                                    nadir_iter_surrogate = np.max(surrogate_objs_temp, axis=0)

                                    for approach,count in zip(approaches,range(approaches_length)):                
                                        set_sizes_all[count] = eh_obj.hypercube_size(reference_point_iter,evaluated_data[iter][approach],obj,nadir_iter)
                                        set_sizes_all_surrogate[count] = eh_obj_surrogate.hypercube_size(reference_point_iter,run_data[iter][approach].objectives,obj,nadir_iter_surrogate)
                                        approach_transformer= Normalizer().fit(evaluated_data[iter][approach])
                                        underlying_objs_norm_approach = approach_transformer.transform(evaluated_data[iter][approach])                                       
                                        
                                        for approach2 in approaches:
                                            if approach != approach2:
                                                approach_transformer_other = Normalizer().fit(evaluated_data[iter][approach2])
                                                norm_approach_other = approach_transformer_other.transform(evaluated_data[iter][approach2])
                                                if underlying_objs_norm_other is None:
                                                    underlying_objs_norm_other = norm_approach_other
                                                else:
                                                    underlying_objs_norm_other = np.vstack((underlying_objs_norm_other,norm_approach_other))
                                        rhv_iters[iter][approach] = rmetric.calc(underlying_objs_norm_approach, others=underlying_objs_norm_other)
                                    
                                    eh_approaches_temp = eh_obj.area(set_sizes_all)
                                    eh_approaches_temp_surrogate = eh_obj_surrogate.area(set_sizes_all_surrogate)
                                    mean_eh_iters = np.add(mean_eh_iters, eh_approaches_temp)

                                    for approach,count in zip(approaches,range(approaches_length)): 
                                        eh_iters[iter][approach] = eh_approaches_temp[count]
                                        eh_iters_surrogate[iter][approach] = eh_approaches_temp_surrogate[count]                                     

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

                                median_eh = np.zeros(approaches_length)
                                median_eh_surrogate = np.zeros(approaches_length)
                                median_rmse = np.zeros(approaches_length)
                                median_rhv = np.zeros(approaches_length)
                                sum_eh = np.zeros(approaches_length)
                                sum_rmse = np.zeros(approaches_length)
                                sum_rhv = np.zeros(approaches_length)
                                for approach,count in zip(approaches,range(approaches_length)):
                                    rmse_stack = np.zeros(len(run_data))
                                    eh_stack = np.zeros(len(run_data))
                                    eh_stack_surrogate = np.zeros(len(run_data))
                                    rhv_stack = np.zeros(len(run_data))
                                    for iter in range(len(run_data)):
                                        eh_stack[iter] = eh_iters[iter][approach]
                                        eh_stack_surrogate[iter] = eh_iters_surrogate[iter][approach]
                                        rmse_stack[iter] = rmse_mv_iters[iter][approach]
                                        rhv_stack[iter] = rhv_iters[iter][approach]
                                    median_eh[count] = np.median(eh_stack)
                                    median_eh_surrogate[count] = np.median(eh_stack_surrogate)
                                    median_rmse[count] = np.median(rmse_stack)
                                    median_rhv[count] = np.median(rhv_stack)
                                    sum_eh[count] = np.sum(eh_stack)
                                    sum_rmse[count] = np.sum(rmse_stack)
                                    sum_rhv[count] = np.sum(rhv_stack) 
                                #return [mean_rmse_mv_iters,mean_eh_iters]
                                #return [median_rmse,median_eh]
                                #return [median_rmse,median_eh_surrogate]
                                #return [median_rmse,median_rhv]
                                #return [sum_rmse,sum_eh]
                                return [median_rmse, median_eh, median_rhv, sum_rmse, sum_eh, sum_rhv]

                            try:
                                temp = Parallel(n_jobs=pool_size)(delayed(parallel_execute)(run, path_to_file, prob, obj) for run in range(nruns))
                                temp = np.asarray(temp)
                                #print(temp)
                                mean_rmse_mv = temp[:, 0]
                                mean_eh = temp[:, 1]
                                mean_rhv = temp[:,2]
                                sum_rmse = temp[:,3]
                                sum_eh = temp[:,4]
                                sum_rhv = temp[:,5]
                                
                                #print(mean_eh)
                                #print(mean_rmse_mv)

                                ax = fig.add_subplot(131)
                                bp = ax.boxplot(mean_rmse_mv, showfliers=False, widths=0.45)
                                #ax.set_title(metric + '_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars))
                                ax.set_title('MVRMSE median comparison')
                                ax.set_xlabel('Approaches')
                                #ax.set_ylabel(metric)
                                ax.set_ylabel('MVRMSE')
                                ax.set_xticklabels(approaches_labels, rotation=75, fontsize=10)

                                ax = fig.add_subplot(132)
                                bp = ax.boxplot(mean_eh, showfliers=False, widths=0.45)
                                #ax.set_title(metric + '_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars))
                                ax.set_title('EH median comparison')
                                ax.set_xlabel('Approaches')
                                #ax.set_ylabel(metric)
                                ax.set_ylabel('EH')
                                ax.set_xticklabels(approaches_labels, rotation=75, fontsize=10)

                                ax = fig.add_subplot(133)
                                bp = ax.boxplot(mean_rhv, showfliers=False, widths=0.45)
                                #ax.set_title(metric + '_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars))
                                ax.set_title('RHV median comparison')
                                ax.set_xlabel('Approaches')
                                #ax.set_ylabel(metric)
                                ax.set_ylabel('RHV')
                                ax.set_xticklabels(approaches_labels, rotation=75, fontsize=10)

                                filename_fig = data_folder + '/test_runs/' + main_directory + '/' + 'MVRMSE_EH' + '_' + str(sample_size) + '_' + samp + '_' + algo + '_' + prob + '_' + str(
                                        obj) + '_' + str(n_vars)
                                
                                if save_fig == 'png':
                                    fig.savefig(filename_fig + '.png', bbox_inches='tight')
                                else:
                                    fig.savefig(filename_fig + '.pdf', bbox_inches='tight')
                                ax.clear()


                                lenx = np.zeros(int(math.factorial(mode_length)/((math.factorial(mode_length-2))*2)))
                                
                                p_value_rmse =  copy.deepcopy(lenx)
                                p_value_eh = copy.deepcopy(lenx)
                                p_value_rmse_sum = copy.deepcopy(lenx)
                                p_value_eh_sum = copy.deepcopy(lenx)
                                p_value_rhv_mean = copy.deepcopy(lenx)
                                p_value_rhv_sum = copy.deepcopy(lenx)

                                p_cor_temp_eh =  copy.deepcopy(lenx)
                                p_cor_temp_rmse = copy.deepcopy(lenx)
                                p_cor_temp_eh_sum = copy.deepcopy(lenx)
                                p_cor_temp_rmse_sum = copy.deepcopy(lenx)
                                p_cor_temp_rhv_mean = copy.deepcopy(lenx)
                                p_cor_temp_rhv_sum = copy.deepcopy(lenx)

                                count = 0
                                count_prev = 0
                                for i in range(mode_length-1):
                                    for j in range(i+1,mode_length):
                                        w, p1 = wilcoxon(x=mean_eh[:, i], y=mean_eh[:, j])
                                        p_value_eh[count] = p1
                                        w, p2 = wilcoxon(x=mean_rmse_mv[:, i], y=mean_rmse_mv[:, j])
                                        p_value_rmse[count] = p2
                                        w, p3 = wilcoxon(x=sum_rmse[:, i], y=sum_rmse[:, j])
                                        p_value_rmse_sum[count] = p3
                                        w, p4 = wilcoxon(x=sum_eh[:, i], y=sum_eh[:, j])
                                        p_value_eh_sum[count] = p4
                                        w, p5 = wilcoxon(x=sum_rhv[:, i], y=sum_rhv[:, j])
                                        p_value_rhv_sum[count] = p5
                                        w, p6 = wilcoxon(x=mean_rhv[:, i], y=mean_rhv[:, j])
                                        p_value_rhv_mean[count] = p6
                                        count +=1
                                    if perform_bonferonni is True:
                                        if mod_p_val is True:
                                            r, p_cor_temp_eh[count_prev:count], alps, alpb = multipletests(p_value_eh[count_prev:count], alpha=0.05, method='bonferroni',
                                                                                is_sorted=False, returnsorted=False)
                                            r, p_cor_temp_rmse[count_prev:count], alps, alpb = multipletests(p_value_rmse[count_prev:count], alpha=0.05, method='bonferroni',
                                                                                is_sorted=False, returnsorted=False)
                                            r, p_cor_temp_eh_sum[count_prev:count], alps, alpb = multipletests(p_value_eh_sum[count_prev:count], alpha=0.05, method='bonferroni',
                                                                                is_sorted=False, returnsorted=False)
                                            r, p_cor_temp_rmse_sum[count_prev:count], alps, alpb = multipletests(p_value_rmse_sum[count_prev:count], alpha=0.05, method='bonferroni',
                                                                                is_sorted=False, returnsorted=False)
                                            r, p_cor_temp_rhv_mean[count_prev:count], alps, alpb = multipletests(p_value_rhv_mean[count_prev:count], alpha=0.05, method='bonferroni',
                                                                                is_sorted=False, returnsorted=False)
                                            r, p_cor_temp_rhv_sum[count_prev:count], alps, alpb = multipletests(p_value_rhv_sum[count_prev:count], alpha=0.05, method='bonferroni',
                                                                                is_sorted=False, returnsorted=False)
                                if perform_bonferonni is True:
                                    p_cor_eh = p_cor_temp_eh
                                    p_cor_rmse = p_cor_temp_rmse
                                    p_cor_eh_sum = p_cor_temp_eh_sum
                                    p_cor_rmse_sum = p_cor_temp_rmse_sum
                                    p_cor_rhv = p_cor_temp_rhv_mean
                                    p_cor_rhv_sum = p_cor_temp_rhv_sum
                                current_index = [sample_size, samp, prob, obj, n_vars]

                                ranking_eh = calc_rank(p_cor_eh, np.median(mean_eh, axis=0),mode_length)
                                ranking_eh_sum = calc_rank(p_cor_eh_sum, np.median(sum_eh, axis=0),mode_length)
                                ranking_rmse = calc_rank(p_cor_rmse, np.median(mean_rmse_mv, axis=0)*-1,mode_length)
                                ranking_rmse_sum = calc_rank(p_cor_rmse_sum, np.median(sum_rmse, axis=0)*-1,mode_length)
                                ranking_rhv = calc_rank(p_cor_rhv, np.median(mean_rhv, axis=0),mode_length)
                                ranking_rhv_sum = calc_rank(p_cor_rhv_sum, np.median(sum_rhv, axis=0),mode_length)

                                p_cor_eh = np.hstack((p_cor_eh, np.mean(mean_eh, axis=0),
                                                            np.median(mean_eh, axis=0),
                                                            np.std(mean_eh, axis=0), ranking_eh))
                                p_cor_eh = np.hstack((current_index, p_cor_eh))

                                p_cor_eh_sum = np.hstack((p_cor_eh_sum, np.mean(sum_eh, axis=0),
                                                            np.median(sum_eh, axis=0),
                                                            np.std(sum_eh, axis=0), ranking_eh_sum))
                                p_cor_eh_sum = np.hstack((current_index, p_cor_eh_sum))

                                p_cor_rmse = np.hstack((p_cor_rmse, np.mean(mean_rmse_mv, axis=0),
                                                            np.median(mean_rmse_mv, axis=0),
                                                            np.std(mean_rmse_mv, axis=0), ranking_rmse))
                                p_cor_rmse = np.hstack((current_index, p_cor_rmse))
                            
                                p_cor_rmse_sum = np.hstack((p_cor_rmse_sum, np.mean(sum_rmse, axis=0),
                                                            np.median(sum_rmse, axis=0),
                                                            np.std(sum_rmse, axis=0), ranking_rmse_sum))
                                p_cor_rmse_sum = np.hstack((current_index, p_cor_rmse_sum))

                                p_cor_rhv = np.hstack((p_cor_rhv, np.mean(mean_rhv, axis=0),
                                                            np.median(mean_rhv, axis=0),
                                                            np.std(mean_rhv, axis=0), ranking_rhv))
                                p_cor_rhv = np.hstack((current_index, p_cor_rhv))

                                p_cor_rhv_sum = np.hstack((p_cor_rhv_sum, np.mean(sum_rhv, axis=0),
                                                            np.median(sum_rhv, axis=0),
                                                            np.std(sum_rhv, axis=0), ranking_rhv_sum))
                                p_cor_rhv_sum = np.hstack((current_index, p_cor_rhv_sum))                                




                                if p_vals_all_eh is None:
                                    p_vals_all_eh = p_cor_eh
                                    p_vals_all_eh_sum = p_cor_eh_sum
                                    p_vals_all_rmse = p_cor_rmse
                                    p_vals_all_rmse_sum = p_cor_rmse_sum
                                    p_vals_all_rhv = p_cor_rhv
                                    p_vals_all_rhv_sum = p_cor_rhv_sum
                                else:
                                    p_vals_all_eh = np.vstack((p_vals_all_eh, p_cor_eh))
                                    p_vals_all_eh_sum = np.vstack((p_vals_all_eh_sum, p_cor_eh_sum))
                                    p_vals_all_rmse = np.vstack((p_vals_all_rmse, p_cor_rmse))
                                    p_vals_all_rmse_sum = np.vstack((p_vals_all_rmse_sum, p_cor_rmse_sum))
                                    p_vals_all_rhv = np.vstack((p_vals_all_rhv, p_cor_rhv))
                                    p_vals_all_rhv_sum = np.vstack((p_vals_all_rhv_sum, p_cor_rhv_sum))
                            except Exception as e:
                                print(str(e)+traceback.format_exc()) 


if mod_p_val is False:
    file_summary = data_folder + '/test_runs/' + main_directory + '/Summary_EH_' + problem_testbench + '.csv'
else:
    file_summary = data_folder + '/test_runs/' + main_directory + '/Summary_EH_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_eh)
writeFile.close()

if mod_p_val is False:
    file_summary = data_folder + '/test_runs/'+ main_directory + '/Summary_RMSE_' + problem_testbench + '.csv'
else:
    file_summary = data_folder + '/test_runs/'+ main_directory + '/Summary_RMSE_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_rmse)
writeFile.close()

if mod_p_val is False:
    file_summary = data_folder + '/test_runs/' + main_directory + '/Summary_EH_sum' + problem_testbench + '.csv'
else:
    file_summary = data_folder + '/test_runs/' + main_directory + '/Summary_EH_sum_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_eh_sum)
writeFile.close()

if mod_p_val is False:
    file_summary = data_folder + '/test_runs/'+ main_directory + '/Summary_RMSE_sum' + problem_testbench + '.csv'
else:
    file_summary = data_folder + '/test_runs/'+ main_directory + '/Summary_RMSE_sum_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_rmse_sum)
writeFile.close()

if mod_p_val is False:
    file_summary = data_folder + '/test_runs/' + main_directory + '/Summary_RHV_' + problem_testbench + '.csv'
else:
    file_summary = data_folder + '/test_runs/' + main_directory + '/Summary_RHV_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_rhv)
writeFile.close()

if mod_p_val is False:
    file_summary = data_folder + '/test_runs/' + main_directory + '/Summary_RHV_sum' + problem_testbench + '.csv'
else:
    file_summary = data_folder + '/test_runs/' + main_directory + '/Summary_RHV_sum_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_rhv_sum)
writeFile.close()


