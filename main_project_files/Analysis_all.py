import sys
sys.path.insert(1, '/mnt/i/AmzNew/')
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

#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('text', usetex=True)
#plt.rcParams.update({'font.size': 15})


#pareto_front_directory = 'True_Pareto_5000'

mod_p_val = True
perform_bonferonni = False
metric = 'HV'
#metric = 'IGD'
save_fig = 'pdf'
#dims = [5,8,10] #,8,10]
dims = [2, 5, 7, 10]
#dims = [5]
sample_sizes = [2000, 10000]#, 50000]
#sample_sizes = [50000]
folder_data = './data/initial_samples'

problem_testbench = 'DTLZ'
#problem_testbench = 'DDMOPP'

#main_directory = 'Offline_Prob_DDMOPP3'
#main_directory = 'Tests_Probabilistic_Finalx_new'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'
#main_directory = 'Test_Gpy3'
#main_directory = 'Test_DR_4'
main_directory = 'Test_DR_CSC_Final_1'

objectives = [3,5,7]
#objectives = [7]
#objectives = [2,3,5]
#objectives = [3,5,6,8,10]

#problems = ['DTLZ2']
#problems = ['DTLZ2','DTLZ4']
#problems = ['P1']
problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2','P3','P4']
#problems = ['P4']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3

#modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
#modes = [0,7,70,71]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [0,1,7,8]
#modes = [0,1,7,8]
#modes = [1,2]
#mode_length = int(np.size(modes))
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




#approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach 3']
#approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach X']
#approaches = ['Initial sampling','Generic', 'Approach Prob','Approach Hybrid']
#approaches = ['Initial sampling','Generic_RVEA','Generic_IBEA']
#approaches = ['Initial sampling','Prob Old', 'Prob constant 1','Prob FE/FEmax']
#approaches = ['Generic', 'Approach Prob','Approach Hybrid']
#approaches = ['Generic', 'Probabilistic','Hybrid']
#approaches = ["generic_fullgp","generic_sparsegp","strategy_2","strategy_3","rf_ne10","rf", "htgp0"]
#approaches = ["generic_fullgp0","generic_fullgp","generic_sparsegp0","generic_sparsegp", "htgp0", "htgp1" , "htgp"]
#approaches = ["generic_fullgp","generic_sparsegp","htgp"]
#approaches = ["generic_fullgp","generic_sparsegp","htgp_1","htgp"]
#approaches = ["generic_fullgp","generic_sparsegp_50","generic_sparsegp","htgp_mse"]
approaches = ["generic_sparsegp","htgp"]

mode_length = int(np.size(approaches))
#approaches = ['7', '9', '11']
#"DTLZ6": {"2": [10, 10], "3": [10, 10, 10], "5": [10, 10, 10, 10, 10]},
#"DTLZ4": {"2": [4, 4], "3": [4, 4, 4], "5": [4, 4, 4, 4, 4]},
#hv_ref = {"DTLZ2": {"2": [3, 3], "3": [3, 3, 3], "5": [3, 3, 3, 3, 3],  "7": [3, 3, 3, 3, 3, 3, 3]},
#          "DTLZ4": {"2": [3, 3.1], "3": [3, 3, 3], "5": [3, 3, 3, 3, 3] ,  "7": [3, 3, 3, 3, 3, 3, 3]},
#          "DTLZ5": {"2": [2.5, 3], "3": [2.5, 3, 3], "5": [2, 2, 2, 2, 2] ,  "7": [3, 3, 3, 3, 3, 3, 3]},
#          "DTLZ6": {"2": [10, 10], "3": [10, 10, 10], "5": [7, 7, 7, 7, 7] ,  "7": [10, 10, 10, 10, 10, 10, 10]},
#          "DTLZ7": {"2": [1, 20], "3": [1, 1, 30], "5": [1, 1, 1, 1, 50] ,  "7": [1, 1, 1, 1, 1, 1, 70]}}

hv_ref = {"DTLZ2": {"2": [3, 3], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6],  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ4": {"2": [3, 3.1], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6] ,  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ5": {"2": [2.5, 3], "3": [6, 6, 6], "5": [6, 6, 6, 6, 6] ,  "7": [6, 6, 6, 6, 6, 6, 6]},
          "DTLZ6": {"2": [10, 10], "3": [20, 20, 20], "5": [20, 20, 20, 20, 20] ,  "7": [20, 20, 20, 20, 20, 20, 20]},
          "DTLZ7": {"2": [1, 20], "3": [1, 1, 40], "5": [1, 1, 1, 1, 50] ,  "7": [1, 1, 1, 1, 1, 1, 70]}}


nruns = 11
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
                        #fig = plt.figure(1, figsize=(10, 10))
                        #fig = plt.figure()
                        #ax = fig.add_subplot(111)
                        #fig.set_size_inches(5, 5)
                        
                        fig = plt.figure()
                        # ax = fig.add_subplot(111, projection='3d')
                        #ax = fig.add_subplot(111)
                        fig.set_size_inches(15, 5)
                        #plt.xlim(0, 1)
                        #plt.ylim(0, 1)

                        #if save_fig == 'pdf':
                        #    plt.rcParams["text.usetex"] = True
                        #with open(pareto_front_directory + '/True_5000_' + prob + '_' + obj + '.txt') as csv_file:
                        #    csv_reader = csv.reader(csv_file, delimiter=',')
                        
                        #############
                        #if problem_testbench is 'DTLZ':
                        #    pareto_front = np.genfromtxt(pareto_front_directory + '/True_5000_' + prob + '_' + str(obj) + '.txt'
                        #                             , delimiter=',')
                        ##############

                        #path_to_file = pareto_front_directory + '/' + 'Pareto_Weld'
                        #infile = open(path_to_file, 'rb')
                        #pareto_front = pickle.load(infile)
                        #infile.close()
                        #problem_weld = WeldedBeam()
                        #pareto_front = problem_weld.pareto_front()

                        for algo in emo_algorithm:
                            #igd_all = np.zeros([nruns, np.shape(modes)[0]])
                            igd_all = None
                            rmse_mv_all = None
                            solution_ratio_all = None
                            time_taken_all = None
                            for mode, mode_count in zip(approaches,range(np.shape(approaches)[0])):

                                    
                                if problem_testbench is 'DTLZ':
                                    """
                                    path_to_file = folder_data + main_directory \
                                            + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                            '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                                            """
                                    path_to_file = './data/test_runs/' + main_directory \
                                                + '/Offline_Mode_' + mode + '_' + algo + \
                                                '/' + samp + '/' + str(sample_size) + '/' + problem_testbench  + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                                else:
                                    path_to_file = './data/test_runs/' + main_directory \
                                                + '/Offline_Mode_' + mode + '_' + algo + \
                                                '/' + samp + '/' + str(sample_size) + '/' + problem_testbench  + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                                print(path_to_file)

                                def igd_calc():
                                    pass

                                def igd_box():
                                    pass

                                def plot_median_run():
                                    pass

                                def plot_convergence_plots():
                                    pass


                                def parallel_execute(run, path_to_file, prob, obj):
                                    rmse_mv_sols = 0
                                    """
                                    if metric is 'IGD':
                                        path_to_file = path_to_file + '/Run_' + str(run) + '_NDS'
                                        infile = open(path_to_file, 'rb')
                                        results_data=pickle.load(infile)
                                        infile.close()
                                        non_dom_front = results_data["actual_objectives_nds"]
                                        non_dom_surr = results_data["surrogate_objectives_nds"]
                                        #print(np.shape(non_dom_surr))
                                        #print((np.max(non_dom_front,axis=0)))
                                        solution_ratio = np.shape(non_dom_front)[0]/np.shape(non_dom_surr)[0]
                                        return [igd(pareto_front, non_dom_front), solution_ratio]
                                    """
                                    #else:
                                    if problem_testbench is 'DDMOPP':
                                        path_to_file = path_to_file + '/Run_' + str(run)
                                        infile = open(path_to_file, 'rb')
                                        results_data=pickle.load(infile)
                                        infile.close()
                                        surrogate_objectives_nds = results_data["obj_solutions"]
                                        time_taken=results_data["time_taken"]
                                        x = []
                                        with open(path_to_file+'_soln','r') as f:
                                            reader = csv.reader(f)
                                            for line in reader: x.append(line)
                                        actual_objectives_nds = x
                                        actual_objectives_nds = np.array(actual_objectives_nds, dtype=np.float32)
                                        ref = [obj*np.sqrt(2)]*obj
                                    else:
                                        """
                                        path_to_file = path_to_file + '/Run_' + str(run) + '_NDS'
                                        infile = open(path_to_file, 'rb')
                                        results_data = pickle.load(infile)
                                        infile.close()
                                        actual_objectives_nds = results_data["actual_objectives_nds"]
                                        #non_dom_surr = results_data["surrogate_objectives_nds"]
                                        """
                                        path_to_file = path_to_file + '/Run_' + str(run)
                                        infile = open(path_to_file, 'rb')
                                        results_data=pickle.load(infile)
                                        infile.close()
                                        surrogate_objectives_nds = results_data["obj_solutions"]
                                        time_taken=results_data["time_taken"]
                                        population = results_data["individuals_solutions"]
                                        probobj = test_problem_builder(
                                                    name=prob, n_of_objectives=obj, n_of_variables=n_vars
                                                )
                                        y = probobj.evaluate(population)[0]
                                        actual_objectives_nds = y
                                        ref = hv_ref[prob][str(obj)]
                                    #ref = hv_ref[prob][str(obj)]
                                    #print(actual_objectives_nds)
                                    if np.shape(actual_objectives_nds)[0] > 1:
                                        non_dom_front = ndx(actual_objectives_nds)
                                        actual_objectives_nds = actual_objectives_nds[non_dom_front[0][0]]
                                    else:
                                        actual_objectives_nds = actual_objectives_nds.reshape(1, obj)
                                    #print(np.shape(actual_objectives_nds))
                                    solution_ratio = 0
                                    hyp = hv(actual_objectives_nds)
                                    hv_x = hyp.compute(ref)
                                    print(np.amax(actual_objectives_nds,axis=0))
                                    for i in range(np.shape(actual_objectives_nds)[0]):
                                        rmse_mv_sols += distance.euclidean(surrogate_objectives_nds[i,:obj],actual_objectives_nds[i,:])
                                    rmse_mv_sols = rmse_mv_sols/np.shape(actual_objectives_nds)[0]

                                    return [hv_x, rmse_mv_sols, time_taken, solution_ratio]


                                temp = Parallel(n_jobs=11)(delayed(parallel_execute)(run, path_to_file, prob, obj) for run in range(nruns))
                                temp=np.asarray(temp)
                                igd_temp = np.transpose(temp[:, 0])
                                solution_ratio_temp = np.transpose(temp[:, 3])
                                rmse_mv_sols_temp = np.transpose(temp[:, 1])
                                time_taken_temp = np.transpose(temp[:, 2])

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
                            """
                            bp = ax.boxplot(solution_ratio_all, showfliers=False)
                            ax.set_title('SOLR_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj))
                            ax.set_xlabel('Approaches')
                            ax.set_ylabel('SOLR')
                            ax.set_xticklabels(approaches, rotation=75, fontsize=8)
                            filename_fig = main_directory + '/SOLR_'+ samp + '_' + algo + '_' + prob + '_' + str(obj) + '.png'
                            fig.savefig(filename_fig, bbox_inches='tight')
                            ax.clear()
                            """""
                            #bp = ax.boxplot(solution_ratio_all, showfliers=False)
                            #filename_fig = main_directory + '/SolnRatio_' + samp + '_' + algo + '_' + prob + '_' + str(obj) + '.png'
                            #fig.savefig(filename_fig, bbox_inches='tight')
                            #ax.clear()


#print(p_vals_all)

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


