import sys
sys.path.insert(1, '/mnt/i/AmzNew/')
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
import pandas as pd

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
import seaborn as sns; sns.set()
#rc('font',**{'family':'serif','serif':['Helvetica']})
#rc('text', usetex=True)
#plt.rcParams.update({'font.size': 15})


#pareto_front_directory = 'True_Pareto_5000'

mod_p_val = True
metric = 'HV'
#metric = 'IGD'
save_fig = 'pdf'
#dims = [5] #,8,10]
dims = [2, 5, 7, 10]
sample_sizes = [2000, 10000, 50000]
folder_data = './data/initial_samples'

#problem_testbench = 'DTLZ'
problem_testbench = 'DDMOPP'

#main_directory = 'Offline_Prob_DDMOPP3'
#main_directory = 'Tests_Probabilistic_Finalx_new'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'
#main_directory = 'Test_Gpy3'
#main_directory = 'Test_DR_4'
main_directory = 'Test_DR_CSC_Final_1'

objectives = [3,5,7]
#objectives = [3]
#objectives = [2,3,5]
#objectives = [3,5,6,8,10]

#problems = ['DTLZ2']
#problems = ['DTLZ2','DTLZ4']
#problems = ['P1']
#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
problems = ['P1','P2','P3','P4']
#problems = ['P1']
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

emo_algorithm = ['RVEA']

approaches = ["htgp"]

mode_length = int(np.size(approaches))

nruns = 11
pool_size = 11

plot_boxplot = True

l = [approaches]*nruns
labels = [list(i) for i in zip(*l)]
labels = [y for x in labels for y in x]

p_vals_all_hv = None
p_vals_all_rmse = None
p_vals_all_time = None
index_all = None

max_length = 50

for sample_size in sample_sizes:
    for samp in sampling:
        for prob in problems:
            for obj in objectives:
                for n_vars in dims:
                    if (problem_testbench == 'DTLZ' and obj < n_vars) or problem_testbench == 'DDMOPP':
                        
                        fig = plt.figure()
                        fig.set_size_inches(15, 5)
                        points_sequence_df = []

                        for algo in emo_algorithm:
                            #igd_all = np.zeros([nruns, np.shape(modes)[0]])
                            igd_all = None
                            rmse_mv_all = None
                            solution_ratio_all = None
                            time_taken_all = None
                            for mode, mode_count in zip(approaches,range(np.shape(approaches)[0])):
                                path_to_file = './data/test_runs/' + main_directory \
                                            + '/Offline_Mode_' + mode + '_' + algo + \
                                            '/' + samp + '/' + str(sample_size) + '/' + problem_testbench  + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                                print(path_to_file)

                                def parallel_execute(run, path_to_file, prob, obj):
                                    path_to_file = path_to_file + '/Run_' + str(run)
                                    infile = open(path_to_file, 'rb')
                                    results_data=pickle.load(infile)
                                    infile.close()
                                    total_points_per_model_sequence = results_data["total_points_per_model_sequence"]
                                    total_points_per_model_sequence = np.asarray(total_points_per_model_sequence)
                                    length_points = np.shape(total_points_per_model_sequence)[0]
                                    if length_points < max_length:                                        
                                        last_value = total_points_per_model_sequence[length_points-1]
                                        print("last value=",last_value)
                                        print(np.shape(total_points_per_model_sequence))
                                        total_points_per_model_sequence= np.vstack((total_points_per_model_sequence,
                                                                                    np.tile(last_value,(max_length-length_points,1))))
                                    print(np.shape(total_points_per_model_sequence))

                                    return total_points_per_model_sequence


                                temp = Parallel(n_jobs=11)(delayed(parallel_execute)(run, path_to_file, prob, obj) for run in range(nruns))
                                temp=np.asarray(temp)

                                for i in range(obj):
                                    for j in range(nruns):
                                        for k in range(max_length):
                                            points_sequence_df.append([k, j, i+1, temp[j, k, i]])
                                points_sequence_dfpd = pd.DataFrame(points_sequence_df, columns=['Iteration', 'Run', 'Objective', 'Number of points'])
                                color_map = plt.cm.get_cmap('viridis')
                                #color_map = color_map(np.linspace(0, 1, obj+1))
                                ax = sns.lineplot(x="Iteration", y="Number of points",
                                                hue="Objective", style="Objective",
                                                markers=True, dashes=False, data=points_sequence_dfpd, palette=color_map)
                                fig = ax.get_figure()
                                fig.show()
                                filename_fig =  './data/test_runs/' + main_directory + '/Points_consumed_progress_' + str(sample_size) + '_' + samp + '_' + algo + '_' + prob + '_' + str(
                                    obj) + '_' + str(n_vars)
                                if save_fig == 'png':
                                    fig.savefig(filename_fig + '.png', bbox_inches='tight')
                                else:
                                    fig.savefig(filename_fig + '.pdf', bbox_inches='tight')
                                ax.clear()

                                



