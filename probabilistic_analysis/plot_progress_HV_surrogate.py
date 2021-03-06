import numpy as np
import pickle
import os
from joblib import Parallel, delayed
import seaborn as sns;

sns.set()
import matplotlib.pyplot as plt
import pandas as pd
import csv
from IGD_calc import igd, igd_plus
from non_domx import ndx
from pygmo import hypervolume as hv
from scipy.spatial import distance
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import math
from ranking_approaches import calc_rank
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from matplotlib import rc
import datetime

# rc('font',**{'family':'serif','serif':['Helvetica']})
# rc('text', usetex=True)
# plt.rcParams.update({'font.size': 15})

sns.set_style("whitegrid")
pareto_front_directory = 'True_Pareto_5000'

mod_p_val = True
metric = 'HV'
# metric = 'IGD'
save_fig = 'pdf'
# dims = [5,8,10] #,8,10]
dims = [10]

folder_data = 'AM_Samples_109_Final'
problem_testbench = 'DTLZ'
#problem_testbench = 'DDMOPP'

# main_directory = 'Offline_Prob_DDMOPP3'
main_directory = '/home/amrzr/Work/Codes/Tests_Probabilistic_Rev2'
#main_directory = 'Tests_CSC_3'
# main_directory = 'Tests_new_adapt'
# main_directory = 'Tests_toys'

objectives = [8]
#objectives = [2,3,5]
#objectives = [2,3,5,8]
#objectives = [2,3,4,5,6,8,10]

#problems = ['DTLZ4']
problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2']
#problems = ['P1']
# problems = ['WELDED_BEAM'] #dims=4
# problems = ['TRUSS2D'] #dims=3

# modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
# modes = [0,7,70,71]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
# modes = [0,1,7,8]
# modes = [0,1,7,8]
#modes = [0, 1, 9, 7, 8,12,72,82]
#modes = [12,72]
#modes = [7,8,82]
modes = [0,9,1,7,8,12,72,82]

mode_length = int(np.size(modes))


# sampling = ['BETA', 'MVNORM']
#sampling = ['LHS']
# sampling = ['BETA','OPTRAND','MVNORM']
# sampling = ['OPTRAND']
#sampling = ['MVNORM']
sampling = ['LHS', 'MVNORM']
#sx='MVNS'
#sx = 'LHS'

# emo_algorithm = ['RVEA','IBEA']
emo_algorithm = ['RVEA']
# emo_algorithm = ['IBEA']
# emo_algorithm = ['NSGAIII']
# emo_algorithm = ['MODEL_CV']


# approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach 3']
# approaches = ['Generic', 'Approach 1', 'Approach 2', 'Approach X']
# approaches = ['Initial sampling','Generic', 'Approach Prob','Approach Hybrid']
# approaches = ['Initial sampling','Generic_RVEA','Generic_IBEA']
# approaches = ['Initial sampling','Prob Old', 'Prob constant 1','Prob FE/FEmax']
# approaches = ['Generic', 'Approach Prob','Approach Hybrid']
# approaches = ['Generic', 'Probabilistic','Hybrid']
# approaches = ['7', '9', '11']
# approaches = ['Init.Samp.','Generic', 'TransferL', 'Probabilistic', 'Hybrid']
#approaches = ['Init', 'Gen', 'TL', 'Prob', 'Hyb','GenMOEAD','ProbMOEAD','HybMOEA/D']
#approaches = ['Init','GenMOEAD','ProbMOEAD']
approaches = ['Init','TL','Gen-RVEA','Prob-RVEA','Hyb-RVEA','Gen-MOEA/D','Prob-MOEA/D','Hyb-MOEA/D']
# "DTLZ6": {"2": [10, 10], "3": [10, 10, 10], "5": [10, 10, 10, 10, 10]},
# "DTLZ4": {"2": [4, 4], "3": [4, 4, 4], "5": [4, 4, 4, 4, 4]},
hv_ref = {"DTLZ2": {"2": [3, 3], "3": [3, 3, 3], "5": [3, 3, 3, 3, 4], "8": [3, 3, 3, 3, 3, 3, 3, 4]},
          "DTLZ4": {"2": [6, 4], "3": [4.5, 4.5, 4.5], "5": [4, 4, 4, 4, 4], "8": [4, 4, 4, 4, 4, 4, 4, 4]},
          "DTLZ5": {"2": [2.5, 3], "3": [2.5, 3, 3], "5": [2, 2, 2, 3, 3], "8": [2, 2, 2, 2, 2, 2, 2, 2.5]},
          "DTLZ6": {"2": [11, 11], "3": [11, 11, 11], "5": [8.5, 8.5, 8.5, 8.5, 11], "8": [7, 7, 7, 7, 7, 7, 7, 7]},
          "DTLZ7": {"2": [2, 20], "3": [2, 2, 30], "5": [2, 2, 2, 2, 50], "8": [2, 2, 2, 2, 2, 2, 2, 90]}}

nruns = 31
f_eval_limit = 40000
f_evals_per = 2000
f_iters = 20
pool_size = 4

plot_boxplot = True

l = [approaches] * nruns
labels = [list(i) for i in zip(*l)]
labels = [y for x in labels for y in x]

file_exists_check = True
p_vals_all = None
index_all = None
log_time = str(datetime.datetime.now())

def arg_median(a):
    if len(a) % 2 == 1:
        return np.where(a == np.median(a))[0][0]
    else:
        l, r = len(a) // 2 - 1, len(a) // 2
        left = np.partition(a, l)[l]
        right = np.partition(a, r)[r]
        return [np.where(a == left)[0][0], np.where(a == right)[0][0]]


for samp in sampling:
    for prob in problems:
        for obj in objectives:
            for n_vars in dims:
                fig = plt.figure(1, figsize=(6, 3.5))
                # fig = plt.figure()
                ax = fig.add_subplot(111)
                #fig.set_size_inches(5.5, 4)
                fig.set_size_inches(6.5, 4)

                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax = fig.add_subplot(111)
                # fig.set_size_inches(5, 5)
                # plt.xlim(0, 1)
                # plt.ylim(0, 1)

                # if save_fig == 'pdf':
                #    plt.rcParams["text.usetex"] = True
                # with open(pareto_front_directory + '/True_5000_' + prob + '_' + obj + '.txt') as csv_file:
                #    csv_reader = csv.reader(csv_file, delimiter=',')
                if problem_testbench is 'DTLZ' and metric is 'IGD':
                    pareto_front = np.genfromtxt(pareto_front_directory + '/True_5000_' + prob + '_' + str(obj) + '.txt'
                                                 , delimiter=',')

                # path_to_file = pareto_front_directory + '/' + 'Pareto_Weld'
                # infile = open(path_to_file, 'rb')
                # pareto_front = pickle.load(infile)
                # infile.close()
                # problem_weld = WeldedBeam()
                # pareto_front = problem_weld.pareto_front()

                for algo in emo_algorithm:
                    # igd_all = np.zeros([nruns, np.shape(modes)[0]])
                    igd_all = None
                    solution_ratio_all = None
                    hv_progress = None
                    hv_progress_dict = {}
                    hv_progress_df = []
                    hv_df = pd.DataFrame()
                    hyp_start = np.zeros(nruns)
                    filename_fig_check = main_directory + '/' + metric + '_progress_HV_surr_' + samp + '_' + algo + '_' + prob + '_' + str(
                        obj) + '_' + str(n_vars) + '.pdf'
                    if os.path.exists(filename_fig_check) is False or file_exists_check is False:
                        try:                    
                            for mode, mode_count in zip(modes, range(np.shape(modes)[0])):
                                if problem_testbench is 'DTLZ':
                                    path_to_file = main_directory \
                                                + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                                '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                                else:
                                    path_to_file = main_directory \
                                                + '/Offline_Mode_' + str(mode) + '_' + algo + \
                                                '/' + samp + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                                print(path_to_file)
                                with open(main_directory+"/log_"+log_time+".txt", "a") as text_file:
                                    text_file.write("\n"+path_to_file+"__Started___"+str(datetime.datetime.now()))


                                def igd_calc():
                                    pass


                                def igd_box():
                                    pass


                                def plot_median_run():
                                    pass


                                def plot_convergence_plots():
                                    pass


                                def parallel_execute(run, path_to_file):

                                    if metric is 'IGD':
                                        path_to_file = path_to_file + '/Run_' + str(run) + '_NDS'
                                        infile = open(path_to_file, 'rb')

                                        results_data = pickle.load(infile)
                                        infile.close()
                                        non_dom_front = results_data["actual_objectives_nds"]
                                        non_dom_surr = results_data["surrogate_objectives_nds"]
                                        # print(np.shape(non_dom_surr))
                                        # print((np.max(non_dom_front,axis=0)))
                                        solution_ratio = np.shape(non_dom_front)[0] / np.shape(non_dom_surr)[0]
                                        return [igd(pareto_front, non_dom_front), solution_ratio]

                                    else:
                                        if problem_testbench is 'DDMOPP':
                                            soln_all = []
                                            if mode != 0:
                                                path_to_file1 = path_to_file + '/Run_' + str(run+1) + '_surr_all'
                                                with open(path_to_file1, 'r') as f:
                                                    reader = csv.reader(f)
                                                    for line in reader: soln_all.append(line)
                                            else:
                                                path_to_file1 = path_to_file + '/Run_' + str(run) + '_SOLN'
                                                infile = open(path_to_file1, 'rb')
                                                results_data = pickle.load(infile)
                                                infile.close()
                                                soln_all = results_data["actual_objectives_nds"]
                                                print(np.shape(soln_all))
                                                soln_all = np.tile(soln_all, (400, 1))

                                            soln_all = np.array(soln_all, dtype=np.float32)
                                            r0=np.ones(obj)
                                            r1=np.ones(obj)*-1
                                            dx=distance.euclidean(r0,r1)
                                            ref=np.ones(obj)*dx
                                            #ref = [2] * obj
                                            print(ref)

                                        else:
                                            """
                                            path_to_file1 = path_to_file + '/Run_' + str(run + 1) + '_soln'
                                            soln_all = []
                                            with open(path_to_file1, 'r') as f:
                                                reader = csv.reader(f)
                                                for line in reader: soln_all.append(line)
                                            soln_all = np.array(soln_all, dtype=np.float32)
                                            """
                                            soln_all = []
                                            if mode != 0:
                                                if mode == 9:
                                                    path_to_file1 = path_to_file + '/Run_' + str(run + 1) + '_surr_all'
                                                    with open(path_to_file1, 'r') as f:
                                                        reader = csv.reader(f)
                                                        for line in reader: soln_all.append(line)
                                                else:
                                                    path_to_filex = path_to_file + '/Run_' + str(run)
                                                    infile = open(path_to_filex, 'rb')
                                                    results_data = pickle.load(infile)
                                                    infile.close()
                                                    soln_all = results_data["objectives_archive"]
                                                    if type(soln_all) is dict:
                                                        soln_all=np.vstack(soln_all.values())    
                                            else:
                                                path_to_file1 = path_to_file + '/Run_' + str(run) + '_NDS'
                                                infile = open(path_to_file1, 'rb')
                                                results_data = pickle.load(infile)
                                                infile.close()
                                                soln_all = results_data["actual_objectives_nds"]
                                                print(np.shape(soln_all))
                                                soln_all = np.tile(soln_all, (400, 1))

                                            soln_all = np.array(soln_all, dtype=np.float32)
                                            ref = hv_ref[prob][str(obj)]

                                        # print(actual_objectives_nds)
                                        hv_iter = []
                                        print(np.shape(soln_all))
                                        max_iter = np.min((np.shape(soln_all)[0], int(f_eval_limit)))
                                        f_evals_per = int(np.shape(soln_all)[0] / f_iters)
                                        # for f_iter in range(int(max_iter / f_evals_per)):
                                        for f_iter in range(f_iters):
                                            # soln_all_temp = soln_all[f_iter * f_evals_per:(f_iter + 1) * f_evals_per, :]
                                            if f_iter == 0:
                                                soln_all_temp = soln_all[0:100, :]
                                            else:
                                                soln_all_temp = soln_all[f_iter * f_evals_per:(f_iter + 1) * f_evals_per, :]
                                            if np.shape(soln_all_temp)[0] > 1:
                                                non_dom_front = ndx(soln_all_temp)
                                                soln_iter = soln_all_temp[non_dom_front[0][0]]
                                            else:
                                                soln_iter = soln_all_temp.reshape(1, obj)
                                            hyp = hv(soln_iter)
                                            hyp_temp = hyp.compute(ref)
                                            """
                                            if f_iter == 0 and mode == 0:
                                                hyp_start[run] = hyp_temp
                                                print(hyp_start[run])
                                            if f_iter == 0 and mode == 9:
                                                hyp_temp = hyp_start[run]
                                                print(hyp_temp)
                                            print(hyp_start)
                                            """
                                            #    hv_iter = hyp_temp
                                            # else:
                                            #   hv_iter = np.vstack((hv_iter, hyp_temp))
                                            hv_iter = np.append(hv_iter, hyp_temp)
                                        # print(hv_iter)
                                        return hv_iter


                                temp = Parallel(n_jobs=pool_size)(delayed(parallel_execute)(run, path_to_file) for run in range(nruns))

                                #temp = None
                                #for run in range(nruns):
                                #   temp = np.append(temp, parallel_execute(run, path_to_file))

                                temp = np.asarray(temp)

                                if hv_progress is None:
                                    hv_progress = [temp]
                                else:
                                    hv_progress = np.vstack((hv_progress, [temp]))
                                # igd_temp = np.transpose(temp[:, 0])
                                for i in range(nruns):
                                    # for j in range(int(f_eval_limit / f_evals_per)):
                                    for j in range(f_iters):
                                        if j == 0 and mode == 0:
                                            hyp_start[i] = temp[i, j]
                                        if j == 0 and mode == 9:
                                            temp[i, j] = hyp_start[i]
                                        hv_progress_df.append([j * f_evals_per, i, approaches[mode_count], temp[i, j]])

                            hv_df = pd.DataFrame(hv_progress_df, columns=['Evaluations', 'Runs', 'Approaches', 'HV (Surrogate)'])

                            print(hv_df)
                            color_map = plt.cm.get_cmap('viridis')
                            color_map = color_map(np.linspace(0, 1, mode_length))

                            ax = sns.lineplot(x="Evaluations", y="HV (Surrogate)",
                                            hue="Approaches", style="Approaches",
                                            markers=True, dashes=False, data=hv_df, palette=color_map)
                            # box = ax.get_position()
                            handles, labels = ax.get_legend_handles_labels()
                            # ax.legend(handles=handles, labels=labels)
                            #ax.legend(handles=handles[1:], labels=labels[1:], frameon=False, loc='upper center',
                            #          bbox_to_anchor=(0.5, 1.15), ncol=mode_length)

                            #ax.legend(handles=handles, labels=labels, frameon=False, loc='upper center',
                            #        bbox_to_anchor=(0.5, 1.15), ncol=mode_length)
                            ax.legend(handles=handles, labels=labels, frameon=False, loc='upper center',
                                                        bbox_to_anchor=(0.5, 1.2), ncol=int(mode_length/2))

                            
                            
                            # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

                            #ax.set_title(prob + ', ' + str(obj) + ' objs, ' + sx)

                            # ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])  # resize position

                            # Put a legend to the right side
                            # ax.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
                            # ax.set_title(prob+'_'+str(obj))
                            fig = ax.get_figure()
                            fig.show()
                            filename_fig = main_directory + '/' + metric + '_progress_HV_surr_' + samp + '_' + algo + '_' + prob + '_' + str(
                                obj) + '_' + str(n_vars)
                            if save_fig == 'png':
                                fig.savefig(filename_fig + '.png', bbox_inches='tight')
                            else:
                                fig.savefig(filename_fig + '.pdf', bbox_inches='tight')
                            ax.clear()
                            plt.clf()
                        except Exception as e:
                            print(str(e))
                            with open(main_directory+"/log_"+log_time+".txt", "a") as text_file:
                                text_file.write("\n"+path_to_file+"_____failed___"+str(e)+"____"+str(datetime.datetime.now()))
                    else:
                        print(filename_fig_check + " ---- already exists")
                        ax.clear()
                        plt.clf()