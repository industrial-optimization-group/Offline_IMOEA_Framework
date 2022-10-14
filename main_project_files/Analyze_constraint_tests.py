import numpy as np
import pickle
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import csv
from other_tools.IGD_calc import igd, igd_plus
from other_tools.non_domx import ndx
from pygmo import hypervolume as hv
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import math
from other_tools.ranking_approaches import  calc_rank
from scipy.spatial import distance
#from pymop.problems.welded_beam import WeldedBeam
#from pymop.problems.truss2d import Truss2D
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import pandas as pd
import copy
#from matplotlib import rc

#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('text', usetex=True)
#plt.rcParams.update({'font.size': 15})


pareto_front_directory = 'True_Pareto_5000'

mod_p_val = True
metric = 'HV'
#metric = 'IGD'
save_fig = 'pdf'
perform_bonferonni = True
plot_scatter = False
#plot_scatter = True

data_folder = '/home/amrzr/Work/Codes/data'
results_folder = 'Test_constrained_4'
init_folder = data_folder + '/constraint_handling_dataset'

problem_testbench = 'DTLZ'
#problem_testbench = 'DBMOPP'

file_instances = init_folder + '/test_instances2.csv'
#file_instances = init_folder + '/test_instances_DBMOPP.csv'


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
#approaches = ["genericRVEA","probRVEA","probRVEA_constraint_v1","probRVEA_constraint_v2"]
approaches = ["genericRVEA","probRVEA","probRVEA_constraint_v2"]
no_of_approaches = len(approaches)

size_instance = np.size(all_problems)

interactive = False

#"DTLZ6": {"2": [10, 10], "3": [10, 10, 10], "5": [10, 10, 10, 10, 10]},
#"DTLZ4": {"2": [4, 4], "3": [4, 4, 4], "5": [4, 4, 4, 4, 4]},
hv_ref = {"DTLZ2": {"2": [2, 2], "3": [3, 3, 3], "5": [3, 3, 3, 3, 3], "7": [3, 3, 3, 3, 3, 3, 3]},
          "DTLZ4": {"2": [3, 3.1], "3": [3, 3, 3], "5": [3, 3, 3, 3, 3], "7": [3, 3, 3, 3, 3, 3, 3]},
          "DTLZ5": {"2": [2.5, 3], "3": [2.5, 3, 3], "5": [2, 2, 2, 2, 2.5], "7": [2, 2, 2, 2, 2, 2, 2.5]},
          "DTLZ6": {"2": [10, 10], "3": [10, 10, 10], "5": [7, 7, 7, 7, 7], "7": [7, 7, 7, 7, 7, 7, 7]},
          "DTLZ7": {"2": [1, 20], "3": [1, 1, 30], "5": [1, 1, 1, 1, 45], "7": [1, 1, 1, 1, 1, 1, 75]}}


nruns = 11
pool_size = 3

plot_boxplot = True

l = [approaches]*nruns
labels = [list(i) for i in zip(*l)]
labels = [y for x in labels for y in x]

p_vals_all_hv = None
p_vals_all_rmse = None
p_vals_all_time = None
index_all = None

def arg_median(a):
    if len(a) % 2 == 1:
        return np.where(a == np.median(a))[0][0]
    else:
        l,r = len(a) // 2 - 1, len(a) // 2
        left = np.partition(a, l)[l]
        right = np.partition(a, r)[r]
        return [np.where(a == left)[0][0], np.where(a == right)[0][0]]

for instance in range(size_instance):
    prob = all_problems[instance]
    n_vars = all_n_vars[instance]
    obj = all_objs[instance]
    sample_size = all_sample_size[instance]
    if problem_testbench == 'DTLZ':
        ndim = all_ndim[instance]
        bound_valL = all_bound_valL[instance]
        bound_valU = all_bound_valU[instance]
    else:
        n_global_pareto_regions = all_n_global_pareto_regions[instance]
        constraint_type = all_constraint_type[instance]        
    
    #fig = plt.figure(1, figsize=(10, 10))
    fig = plt.figure()
    #ax = fig.add_subplot(111)
    fig.set_size_inches(15, 5)
    
    igd_all = None
    rmse_all = None
    success_ratio_all = None
    solution_ratio_all = None
    for approach, approach_count in zip(approaches,range(np.shape(approaches)[0])):
        if problem_testbench == 'DTLZ':
            problem_spec = prob +'_'+ str(sample_size) + '_' + str(obj) + '_' + \
                            str(n_vars) +  '_b'+str(ndim) +'_' + str(bound_valL).replace('.','') + \
                                str(bound_valU).replace('.','')
        else:
            problem_spec = prob +'_'+ str(sample_size) + '_' + str(obj) + '_' + \
                            str(n_vars) +  '_b'+str(n_global_pareto_regions) +'_' + str(constraint_type)

        path_to_folder = data_folder + '/test_runs/'+  results_folder \
                    + '/' + approach + '/' + problem_spec
        print(path_to_folder)

        def parallel_execute(run):
            rmse_mv_sols = 0
            file_success = path_to_folder + '/Run_'+ str(run) + '_data_success.csv'
            data_success = pd.read_csv(file_success)
            file_class = path_to_folder + '/Run_' + str(run) + '_data_class.csv'
            data_class = pd.read_csv(file_class)
            file_surrogate = path_to_folder + '/Run_'+ str(run) + '_data_surrogate.csv'
            data_surrogate = pd.read_csv(file_surrogate)
            file_all = path_to_folder + '/Run_'+ str(run) + '_data_all.csv'
            data_all = pd.read_csv(file_all)
            x_data_success = data_success.values[:,0:n_vars]
            y_data_success = data_success.values[:,n_vars:n_vars+obj]
            y_data_surrogate = data_surrogate.values[:,n_vars:n_vars+obj]
            success_locs = data_surrogate.values[:,n_vars+obj:n_vars+obj+1]
            y_data_surrogate_success = y_data_surrogate[np.where(success_locs==1)[0]]
            if metric == 'IGD':
                pass
                """
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

            else:
                if problem_testbench == 'DBMOPP':
                    r0=np.ones(obj)
                    r1=np.ones(obj)*-1
                    dx=distance.euclidean(r0,r1)
                    ref=np.ones(obj)*dx
                else:
                    ref = hv_ref[prob][str(obj)]
                print("No. of successful solutions:", np.shape(y_data_success)[0])
                if np.shape(y_data_success)[0] > 1:
                    non_dom_front = ndx(y_data_success)
                    actual_objectives_nds = y_data_success[non_dom_front[0][0]]
                elif np.shape(y_data_success)[0] == 1:
                    actual_objectives_nds = y_data_success.reshape(1, obj)
                else:
                    actual_objectives_nds = None
                    hv_x = 0
                    success_ratio_temp = 0

                solution_ratio = 0
                if actual_objectives_nds is not None:
                    print("No. of non-dominated solutions:", np.shape(actual_objectives_nds)[0])                    
                    hyp = hv(actual_objectives_nds)
                    hv_x = hyp.compute(ref)
                    print("Hypervolume:",hv_x)
                    if plot_scatter:
                        fig1 = plt.figure()
                        fig1.scatter(y_data_success[:,0],y_data_success[:,1],c='blue')
                        fig1.scatter(actual_objectives_nds[:,0], actual_objectives_nds[:,1], c='red')
                        filename_figx = data_folder + '/test_runs/'+  results_folder \
                        + '/plots/' + problem_spec + '_' + approach + '_' + str(run)
                        fig1.savefig(filename_figx + '.pdf', bbox_inches='tight')
                        #ax.clear()
                
                    ##### RMSE
                    for i in range(np.shape(y_data_surrogate_success)[0]):
                        rmse_mv_sols += distance.euclidean(y_data_surrogate_success[i,:],y_data_success[i,:])
                    rmse_mv_sols = rmse_mv_sols/np.shape(y_data_surrogate_success)[0]
                    print("MV-RMSE:", rmse_mv_sols)
                    ##### Success ratio
                    success_ratio_temp = np.shape(y_data_surrogate_success)[0] / np.shape(y_data_surrogate)[0]
                    print("Success ratio:", success_ratio_temp)

                return [hv_x, solution_ratio, rmse_mv_sols, success_ratio_temp]



        temp = Parallel(n_jobs=pool_size)(delayed(parallel_execute)(run) for run in range(nruns))
        #temp=None
        #for run in range(nruns):
        #    temp=np.append(temp,parallel_execute(run, path_to_file))

        temp=np.asarray(temp)
        igd_temp = np.transpose(temp[:, 0])
        solution_ratio_temp = np.transpose(temp[:, 1])
        rmse_temp = np.transpose(temp[:, 2])
        success_ratio_temp = np.transpose(temp[:, 3])

        if plot_boxplot is True:
            if igd_all is None:
                igd_all = igd_temp
                solution_ratio_all = solution_ratio_temp
                rmse_all = rmse_temp
                success_ratio_all = success_ratio_temp
            else:
                igd_all = np.vstack((igd_all, igd_temp))
                solution_ratio_all = np.vstack((solution_ratio_all,solution_ratio_temp))
                rmse_all = np.vstack((rmse_all,rmse_temp))
                success_ratio_all = np.vstack((success_ratio_all, success_ratio_temp))
    igd_all = np.transpose(igd_all)
    solution_ratio_all = np.transpose(solution_ratio_all)
    rmse_all = np.transpose(rmse_all)
    success_ratio_all = np.transpose(success_ratio_all)

    lenx = np.zeros(int(math.factorial(no_of_approaches)/((math.factorial(no_of_approaches-2))*2)))
    p_value_rmse =  copy.deepcopy(lenx)
    p_value_hv = copy.deepcopy(lenx)
    p_value_time = copy.deepcopy(lenx)
    p_cor_temp_hv =  copy.deepcopy(lenx)
    p_cor_temp_rmse = copy.deepcopy(lenx)
    p_cor_temp_time = copy.deepcopy(lenx)
    count = 0
    count_prev = 0
    for i in range(no_of_approaches-1):
        for j in range(i+1,no_of_approaches):
            w, p1 = wilcoxon(x=igd_all[:, i], y=igd_all[:, j])
            p_value_hv[count] = p1
            w, p2 = wilcoxon(x=rmse_all[:, i], y=rmse_all[:, j])
            p_value_rmse[count] = p2
            try:
                w, p3 = wilcoxon(x=success_ratio_all[:, i], y=success_ratio_all[:, j])
                p_value_time[count] = p3
            except Exception as e:
                p_value_time[count] = 1
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
    if problem_testbench == 'DTLZ':
        current_index = [prob, n_vars, obj, sample_size, ndim, bound_valL, bound_valU]
    else:
        current_index = [prob, n_vars, obj, sample_size, n_global_pareto_regions, constraint_type]
    ranking_hv = calc_rank(p_cor_hv, np.median(igd_all, axis=0),no_of_approaches)
    ranking_rmse = calc_rank(p_cor_rmse, np.median(rmse_all, axis=0)*-1,no_of_approaches)
    ranking_time = calc_rank(p_cor_time, np.median(success_ratio_all, axis=0),no_of_approaches)
    #adding other indicators mean, median, std dev
    #p_cor = (np.asarray([p_cor, np.mean(igd_all, axis=0),
    #                             np.median(igd_all, axis=0),
    #                             np.std(igd_all, axis=0)])).flatten()
    p_cor_hv = np.hstack((p_cor_hv, np.mean(igd_all, axis=0),
                                np.median(igd_all, axis=0),
                                np.std(igd_all, axis=0), ranking_hv))
    p_cor_hv = np.hstack((current_index,p_cor_hv))

    p_cor_rmse = np.hstack((p_cor_rmse, np.mean(rmse_all, axis=0),
                                np.median(rmse_all, axis=0),
                                np.std(rmse_all, axis=0), ranking_rmse))
    p_cor_rmse = np.hstack((current_index,p_cor_rmse))

    p_cor_time = np.hstack((p_cor_time, np.mean(success_ratio_all, axis=0),
                                np.median(success_ratio_all, axis=0),
                                np.std(success_ratio_all, axis=0), ranking_time))
    p_cor_time = np.hstack((current_index,p_cor_time))

    if p_vals_all_hv is None:
        p_vals_all_hv = p_cor_hv
        p_vals_all_rmse = p_cor_rmse
        p_vals_all_time = p_cor_time
    else:
        p_vals_all_hv = np.vstack((p_vals_all_hv, p_cor_hv))
        p_vals_all_rmse = np.vstack((p_vals_all_rmse, p_cor_rmse))
        p_vals_all_time = np.vstack((p_vals_all_time, p_cor_time))



    """
    p_value = np.zeros(int(math.factorial(no_of_approaches)/((math.factorial(no_of_approaches-2))*2)))
    
    p_cor_temp = p_value
    count = 0
    count_prev = 0
    for i in range(no_of_approaches-1):
        for j in range(i+1,no_of_approaches):
            w, p = wilcoxon(x=igd_all[:, i], y=igd_all[:, j])

            
            p_value[count] = p
            count +=1
        if mod_p_val is True:
            r, p_cor_temp[count_prev:count], alps, alpb = multipletests(p_value[count_prev:count], alpha=0.05, method='bonferroni',
                                                    is_sorted=False, returnsorted=False)
            count_prev = count
    p_cor = p_cor_temp
    print(p_value)
    print(p_cor)
    if mod_p_val is False:
        r, p_cor, alps, alpb = multipletests(p_value, alpha=0.05, method='bonferroni', is_sorted=False,
                                                returnsorted=False)
    current_index = [prob, n_vars, obj, sample_size, ndim, bound_valL, bound_valU]

    #adding other indicators mean, median, std dev
    #p_cor = (np.asarray([p_cor, np.mean(igd_all, axis=0),
    #                             np.median(igd_all, axis=0),
    #                             np.std(igd_all, axis=0)])).flatten()
    ranking = calc_rank(p_cor, np.median(igd_all, axis=0),no_of_approaches)
    p_cor = np.hstack((p_cor, np.mean(igd_all, axis=0),
                                    np.median(igd_all, axis=0),
                                    np.std(igd_all, axis=0),ranking))
    
    """
    
    for medz in range(no_of_approaches):
        print("Median index:")
        med_mode = int(arg_median(igd_all[:, medz]))
        med_mode = np.asarray(med_mode).reshape((1,1))
        med_mode = np.tile(med_mode,(2,2))
        med_mode = med_mode.astype(int)
        path_to_file2 = path_to_folder + '/med_indices.csv'
        print(path_to_file2)
        med_mode = med_mode.tolist()
        print(med_mode)
        with open(path_to_file2, 'w') as f:
            writer = csv.writer(f)
            for line in med_mode: writer.writerow(line)
        print("File written")

    """
    p_cor = np.hstack((current_index,p_cor))
    if p_vals_all is None:
        p_vals_all = p_cor
    else:
        p_vals_all = np.vstack((p_vals_all, p_cor))
    """

    ax = fig.add_subplot(131)
    bp = ax.boxplot(igd_all, showfliers=False, widths=0.45)
    #ax.set_title(metric + '_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars))
    ax.set_title('Hypervolume comparison (higher=better)')
    ax.set_xlabel('Approaches')
    #ax.set_ylabel(metric)
    ax.set_ylabel('Hypervolume')
    ax.set_xticklabels(approaches, rotation=75, fontsize=10)

    ax = fig.add_subplot(132)
    bp = ax.boxplot(rmse_all, showfliers=False, widths=0.45)
    #ax.set_title(metric + '_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars))
    ax.set_title('Accuracy comparison (lower=better)')
    ax.set_xlabel('Approaches')
    #ax.set_ylabel(metric)
    ax.set_ylabel('RMSE_MV')
    ax.set_xticklabels(approaches, rotation=75, fontsize=10)

    print(success_ratio_all)
    ax = fig.add_subplot(133)
    bp = ax.boxplot(success_ratio_all, showfliers=False, widths=0.45)
    #ax.set_title(metric + '_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars))
    ax.set_title('Success ratio (higher=better)')
    ax.set_xlabel('Approaches')
    #ax.set_ylabel(metric)
    ax.set_ylabel('Success ratio')
    ax.set_xticklabels(approaches, rotation=75, fontsize=10)

    filename_fig = data_folder + '/test_runs/'+  results_folder \
                + '/' + problem_spec
    if save_fig == 'png':
        fig.savefig(filename_fig + '.png', bbox_inches='tight')
    else:
        fig.savefig(filename_fig + '.pdf', bbox_inches='tight')
    ax.clear()

    """
    bp = ax.boxplot(igd_all, showfliers=False, widths=0.45)


    #ax.set_title(metric + '_'+ samp + '_Algo_' + algo + '_' + prob + '_' + str(obj) + '_' + str(n_vars))
    ax.set_title('Hypervolume comparison')
    ax.set_xlabel('Approaches')
    #ax.set_ylabel(metric)
    ax.set_ylabel('Hypervolume')
    ax.set_xticklabels(approaches, rotation=45, fontsize=15)
    filename_fig = data_folder + '/test_runs/'+  results_folder \
                    + '/' + problem_spec

    if save_fig == 'png':
        fig.savefig(filename_fig + '.png', bbox_inches='tight')
    else:
        fig.savefig(filename_fig + '.pdf', bbox_inches='tight')
    ax.clear()
    """

if mod_p_val is False:
    file_summary = data_folder + '/test_runs/'+  results_folder + '/Summary_HV_' + problem_testbench + '.csv'
else:
    file_summary = data_folder + '/test_runs/'+  results_folder + '/Summary_HV_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_hv)
writeFile.close()

if mod_p_val is False:
    file_summary = data_folder + '/test_runs/'+  results_folder + '/Summary_RMSE_' + problem_testbench + '.csv'
else:
    file_summary = data_folder + '/test_runs/'+  results_folder + '/Summary_RMSE_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_rmse)
writeFile.close()

if mod_p_val is False:
    file_summary = data_folder + '/test_runs/'+  results_folder + '/Summary_SRATIO_' + problem_testbench + '.csv'
else:
    file_summary = data_folder + '/test_runs/'+  results_folder + '/Summary_SRATIO_ModP_' + problem_testbench + '.csv'
with open(file_summary, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(p_vals_all_time)
writeFile.close()