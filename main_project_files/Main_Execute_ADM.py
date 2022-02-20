import sys
sys.path.insert(1, '/scratch/project_2003769/Codes/Offline_IMOEA_Framework/')


import adm_emo.baseADM
from adm_emo.baseADM import *
import adm_emo.generatePreference as gp


from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.OfflineRVEA import RVEA as RVEA_0
from desdeo_emo.EAs.OfflineRVEA import ProbRVEAv3 as ProbRVEA_0
from desdeo_emo.EAs.OfflineRVEAnew import RVEA as RVEA_1
from desdeo_emo.EAs.OfflineRVEAnew import ProbRVEAv3 as ProbRVEA_1
from desdeo_emo.EAs.NSGAIII import NSGAIII


from desdeo_problem.Problem import DataProblem
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from main_project_files.surrogate_fullGP import FullGPRegressor as fgp

from sklearn.gaussian_process.kernels import Matern
from pyDOE import lhs
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pygmo import non_dominated_front_2d as nd2
from non_domx import ndx
import scipy.io
from sklearn.neighbors import NearestNeighbors
import time
import GPy
from desdeo_problem.testproblems.TestProblems import test_problem_builder

from pymoo.factory import get_problem, get_reference_directions
import adm_emo.rmetric as rm
from sklearn.preprocessing import Normalizer
from pymoo.configuration import Configuration

from pyDOE import lhs
import copy

data_folder = '/scratch/project_2003769/Codes/data'
init_folder = data_folder + '/initial_samples_109'

def compute_nadir(population):
    max_gen = None
    for i in population.objectives_archive:
        if max_gen is None:
            max_gen = np.amax(population.objectives_archive[i], axis=0)
        else:
            max_gen = np.amax(np.vstack((population.objectives_archive[i], max_gen)), axis=0)
    return max_gen


def build_surrogates(problem_testbench, problem_name, nobjs, nvars, nsamples, sampling, is_data, x_data, y_data):
    x_names = [f'x{i}' for i in range(1,nvars+1)]
    y_names = [f'f{i}' for i in range(1,nobjs+1)]
    row_names = ['lower_bound','upper_bound']
    if is_data is False:
        prob = test_problem_builder(problem_name, nvars, nobjs)
        x = lhs(nvars, nsamples)
        y = prob.evaluate(x)
        data = pd.DataFrame(np.hstack((x,y.objectives)), columns=x_names+y_names)
    else:
        data = pd.DataFrame(np.hstack((x_data,y_data)), columns=x_names+y_names)
    if problem_testbench == 'DDMOPP':
        x_low = np.ones(nvars)*-1
        x_high = np.ones(nvars)
    elif problem_testbench == 'DTLZ':
        x_low = np.ones(nvars)*0
        x_high = np.ones(nvars)
    bounds = pd.DataFrame(np.vstack((x_low,x_high)), columns=x_names, index=row_names)
    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names,bounds=bounds)
    start = time.time()
    #problem.train(GaussianProcessRegressor) #, model_parameters={"kernel": Matern(nu=3/2)})
    problem.train(fgp)
    end = time.time()
    time_taken = end - start
    return problem, time_taken

def read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run):
    mat = scipy.io.loadmat(init_folder+'/Initial_Population_' + problem_testbench + '_' + sampling +
                        '_AM_' + str(nvars) + '_'+str(nsamples)+'.mat')
    x = ((mat['Initial_Population_'+problem_testbench])[0][run])[0]
    if problem_testbench == 'DDMOPP':
        mat = scipy.io.loadmat(init_folder+'/Obj_vals_DDMOPP_'+sampling+'_AM_'+problem_name+'_'
                                + str(nobjs) + '_' + str(nvars)
                                + '_'+str(nsamples)+'.mat')
        y = ((mat['Obj_vals_DDMOPP'])[0][run])[0]
    elif problem_testbench == 'DTLZ':
        prob = test_problem_builder(
                    name=problem_name, n_of_objectives=nobjs, n_of_variables=nvars
                )
        y = prob.evaluate(x)[0]
    #elif problem_testbench == 'GAA':
    #    mat = scipy.io.loadmat('./'+folder_data+'/Obj_vals_GAA_'+sampling+'_AM_'+self.name+'_'
    #                            + str(self.num_of_objectives) + '_' + str(self.num_of_variables)
    #                            + '_'+str(sample_size)+'.mat')
    #    y = ((mat['Obj_vals_GAA'])[0][self.run])[0]
    return x, y

def run_adm(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, is_data, approaches, run):

    # ADM parameters
    L = 10  # number of iterations for the learning phase
    D = 5 # number of iterations for the decision phase
    lattice_res_options = [49, 13, 7, 5, 4, 3, 3, 3, 3]
    if nobjs < 11:   # density variable for creating reference vectors
        lattice_resolution = lattice_res_options[nobjs- 2]
    else:
        lattice_resolution = 3
    num_gen_per_iter = 100
    num_approaches = len(approaches)
    dict_moea_objs = {}
    dict_pref_int_moea = {}
    dict_archive_all = {}
    print("Reading data ...")
    x_data, y_data = read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run)
    print("Building surrogates ...")
    problem, time_taken = build_surrogates(problem_testbench, problem_name, nobjs, nvars, nsamples, sampling, is_data, x_data, y_data)

    # define MOEA objects
    for approach in approaches:
        if approach == 'genericRVEA_0':
            dict_moea_objs[approach] = RVEA_0(problem=problem, interact=True, use_surrogates=True, n_gen_per_iter=num_gen_per_iter)
        elif approach == 'probRVEA_0':
            dict_moea_objs[approach] = ProbRVEA_0(problem=problem, interact=True, use_surrogates=True, n_gen_per_iter=num_gen_per_iter)
        elif approach == 'genericRVEA_1':
            dict_moea_objs[approach] = RVEA_1(problem=problem, interact=True, use_surrogates=True, n_gen_per_iter=num_gen_per_iter)
        elif approach == 'probRVEA_1':
            dict_moea_objs[approach] = ProbRVEA_1(problem=problem, interact=True, use_surrogates=True, n_gen_per_iter=num_gen_per_iter)
        elif approach == 'genericNSGAIII':
            dict_moea_objs[approach] = NSGAIII(problem=problem, interact=True, use_surrogates=True, n_gen_per_iter=num_gen_per_iter)

    nadir_data = np.max(y_data,axis=0)
    # initial reference point is specified randomly
    response = np.random.rand(nobjs) + nadir_data
    print("Reference point:", response)
    dict_archive_all[0] = {}
    dict_archive_all[0]['reference_point'] = response
    dict_archive_all[0]['phase'] = 'S'
    for approach in approaches:
        print("Optimizing approach: ", approach)
        # run algorithms once with the randomly generated reference point
        _, pref_int_moea = dict_moea_objs[approach].requests()
        pref_int_moea.response = pd.DataFrame(
            [response], columns=pref_int_moea.content["dimensions_data"].columns
        )
        _, pref_int_moea = dict_moea_objs[approach].iterate(pref_int_moea)
        dict_archive_all[0][approach] = copy.deepcopy(dict_moea_objs[approach].population)

    ##########################
    # change this line later
    # build initial composite front
    cf = generate_composite_front(
        dict_moea_objs[approaches[0]].population.objectives, 
        dict_moea_objs[approaches[1]].population.objectives,
        dict_moea_objs[approaches[2]].population.objectives, 
        dict_moea_objs[approaches[3]].population.objectives, 
        do_nds=False
    )


    # the following two lines for getting pareto front by using pymoo framework
    #ref_dirs = get_reference_directions("das-dennis", nobjs, n_partitions=12)
    # creates uniformly distributed reference vectors
    reference_vectors = ReferenceVectors(lattice_resolution, nobjs)

    # learning phase
    for i in range(L):
        # After this class call, solutions inside the composite front are assigned to reference vectors
        base = baseADM(cf, reference_vectors)
        problem.ideal = np.asarray(base.ideal_point)

        for approach in approaches:
            dict_moea_objs[approach].population.ideal_objective_vector =  np.asarray(base.ideal_point)
            dict_moea_objs[approach].population.ideal_fitness_val =  np.asarray(base.ideal_point)
            #print("Ideal_"+approach+":", dict_moea_objs[approach].population.ideal_objective_vector)

        print("Problem Ideal:",problem.ideal)
        # generates the next reference point for the next iteration in the learning phase
        response = gp.generateRP4learning(base)
        print("Reference point:", response)
        dict_archive_all[i+1] = {}
        dict_archive_all[i+1]['reference_point'] = response
        dict_archive_all[i+1]['phase'] = 'L'
        for approach in approaches:
            print("Optimizing approach: ", approach)
            # run algorithms with the new reference point
            _, pref_int_moea = dict_moea_objs[approach].requests()
            pref_int_moea.response = pd.DataFrame(
                [response], columns=pref_int_moea.content["dimensions_data"].columns
            )
            _, pref_int_moea = dict_moea_objs[approach].iterate(pref_int_moea)
            dict_archive_all[i+1][approach] = copy.deepcopy(dict_moea_objs[approach].population)

        # extend composite front with newly obtained solutions
        cf = generate_composite_front(
            dict_moea_objs[approaches[0]].population.objectives, 
            dict_moea_objs[approaches[1]].population.objectives,
            dict_moea_objs[approaches[2]].population.objectives, 
            dict_moea_objs[approaches[3]].population.objectives, 
            do_nds=False
        )

    # Decision phase
    base = baseADM(cf, reference_vectors)
    # After the learning phase the reference vector which has the maximum number of assigned solutions forms ROI
    max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)

    for i in range(D):

        # since composite front grows after each iteration this call should be done for each iteration
        base = baseADM(cf, reference_vectors)
        problem.ideal = np.asarray(base.ideal_point)

        for approach in approaches:
            dict_moea_objs[approach].population.ideal_objective_vector =  np.asarray(base.ideal_point)
            dict_moea_objs[approach].population.ideal_fitness_val =  np.asarray(base.ideal_point)
            print("Ideal_"+approach+":", dict_moea_objs[approach].population.ideal_objective_vector)
        print("Problem Ideal:",problem.ideal)
        # generates the next reference point for the decision phase
        response = gp.generateRP4decision(base, max_assigned_vector[0])
        print("Reference point:", response)
        dict_archive_all[i+L+1] = {}
        dict_archive_all[i+L+1]['reference_point'] = response
        dict_archive_all[i+L+1]['phase'] = 'D'
        for approach in approaches:
            print("Optimizing approach: ", approach)
            # run algorithms with the new reference point
            _, pref_int_moea = dict_moea_objs[approach].requests()
            pref_int_moea.response = pd.DataFrame(
                [response], columns=pref_int_moea.content["dimensions_data"].columns
            )
            _, pref_int_moea = dict_moea_objs[approach].iterate(pref_int_moea)
            dict_archive_all[i+L+1][approach] = copy.deepcopy(dict_moea_objs[approach].population)

        # extend composite front with newly obtained solutions
        cf = generate_composite_front(
            dict_moea_objs[approaches[0]].population.objectives, 
            dict_moea_objs[approaches[1]].population.objectives,
            dict_moea_objs[approaches[2]].population.objectives, 
            dict_moea_objs[approaches[3]].population.objectives, 
            do_nds=False
        )

    print(dict_archive_all)
    return dict_archive_all