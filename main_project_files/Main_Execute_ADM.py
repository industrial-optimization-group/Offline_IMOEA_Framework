import sys
#sys.path.insert(1, '/scratch/project_2003769/HTGP_MOEA_CSC')
sys.path.insert(1, '/home/amrzr/Work/Codes/AmzNew/')


import adm_emo.baseADM
from adm_emo.baseADM import *
import adm_emo.generatePreference as gp


from desdeo_emo.EAs.RVEA import RVEA
#from desdeo_emo.EAs.OfflineRVEA import RVEA
from desdeo_emo.EAs.OfflineRVEA import ProbRVEAv3
from desdeo_emo.EAs.NSGAIII import NSGAIII


from desdeo_problem.Problem import DataProblem
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from desdeo_problem.testproblems.TestProblems import test_problem_builder

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
    problem.train(GaussianProcessRegressor) #, model_parameters={"kernel": Matern(nu=3/2)})
    end = time.time()
    time_taken = end - start
    return problem, time_taken

def read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run):
    mat = scipy.io.loadmat('./data/initial_samples_old/Initial_Population_' + problem_testbench + '_' + sampling +
                        '_AM_' + str(nvars) + '_'+str(nsamples)+'.mat')
    x = ((mat['Initial_Population_'+problem_testbench])[0][run])[0]
    if problem_testbench == 'DDMOPP':
        mat = scipy.io.loadmat('./data/initial_samples_old/Obj_vals_DDMOPP_'+sampling+'_AM_'+problem_name+'_'
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
    L = 4  # number of iterations for the learning phase
    D = 3  # number of iterations for the decision phase
    lattice_resolution = 5  # density variable for creating reference vectors
    num_gen_per_iter = 5
    num_approaches = len(approaches)
    dict_moea_objs = {}
    dict_pref_int_moea = {}
    dict_archive_all = {}

    x_data, y_data = read_dataset(problem_testbench, problem_name, nobjs, nvars, sampling, nsamples, run)
    problem, time_taken = build_surrogates(problem_testbench, problem_name, nobjs, nvars, nsamples, sampling, is_data, x_data, y_data)

    # define MOEA objects
    for approach in approaches:
        if approach == 'genericRVEA':
            dict_moea_objs[approach] = RVEA(problem=problem, interact=True, use_surrogates=True, n_gen_per_iter=num_gen_per_iter)
        elif approach == 'probRVEA':
            dict_moea_objs[approach] = ProbRVEAv3(problem=problem, interact=True, use_surrogates=True, n_gen_per_iter=num_gen_per_iter)
        elif approach == 'genericNSGAIII':
            dict_moea_objs[approach] = NSGAIII(problem=problem, interact=True, use_surrogates=True, n_gen_per_iter=num_gen_per_iter)


    # initial reference point is specified randomly
    response = np.random.rand(nobjs)
    print("Reference point:", response)
    dict_archive_all[0] = {}
    dict_archive_all[0]['reference_point'] = response
    dict_archive_all[0]['phase'] = 'S'
    for approach in approaches:
        #int_moea_temp = dict_moea_objs[approach]
        # run algorithms once with the randomly generated reference point
        _, pref_int_moea = dict_moea_objs[approach].requests()
        pref_int_moea.response = pd.DataFrame(
            [response], columns=pref_int_moea.content["dimensions_data"].columns
        )
        _, pref_int_moea = dict_moea_objs[approach].iterate(pref_int_moea)
        #dict_moea_objs[approach] = int_moea_temp
        dict_archive_all[0][approach] = dict_moea_objs[approach].population


    # change this line later
    # build initial composite front
    cf = generate_composite_front(
        dict_moea_objs[approaches[0]].population.objectives, 
        dict_moea_objs[approaches[1]].population.objectives, 
        do_nds=False
    )


    # the following two lines for getting pareto front by using pymoo framework
    #problemR = get_problem(problem_name.lower(), nvars, nobjs)
    ref_dirs = get_reference_directions("das-dennis", nobjs, n_partitions=12)
    #pareto_front = problemR.pareto_front(ref_dirs)

    # creates uniformly distributed reference vectors
    reference_vectors = ReferenceVectors(lattice_resolution, nobjs)

    # learning phase
    for i in range(L):
        #data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
        #    problem_name,
        #    n_obj,
        #    i + 1,
        #    gen,
        #]

        # After this class call, solutions inside the composite front are assigned to reference vectors
        base = baseADM(cf, reference_vectors)
        problem.ideal = np.asarray(base.ideal_point)

        for approach in approaches:
            dict_moea_objs[approach].population.ideal_objective_vector =  np.asarray(base.ideal_point)
            print("Ideal_"+approach+":", dict_moea_objs[approach].population.ideal_objective_vector)

        print("Problem Ideal:",problem.ideal)
        # generates the next reference point for the next iteration in the learning phase
        response = gp.generateRP4learning(base)
        print("Reference point:", response)
        dict_archive_all[i+1] = {}
        dict_archive_all[i+1]['reference_point'] = response
        dict_archive_all[i+1]['phase'] = 'L'
        for approach in approaches:
            # run algorithms with the new reference point
            _, pref_int_moea = dict_moea_objs[approach].requests()
            pref_int_moea.response = pd.DataFrame(
                [response], columns=pref_int_moea.content["dimensions_data"].columns
            )
            _, pref_int_moea = dict_moea_objs[approach].iterate(pref_int_moea)
            dict_archive_all[i+1][approach] = dict_moea_objs[approach].population


        # extend composite front with newly obtained solutions
        cf = generate_composite_front(
            dict_moea_objs[approaches[0]].population.objectives, 
            dict_moea_objs[approaches[1]].population.objectives, 
            do_nds=False
        )

        # R-metric calculation
        #ref_point = response.reshape(1, nobjs)

        # normalize reference point
        #rp_transformer = Normalizer().fit(ref_point)
        #norm_rp = rp_transformer.transform(ref_point)

        #rmetric = rm.RMetric(problemR, norm_rp, pf=pareto_front)

        # normalize solutions before sending r-metric
        #rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
        #norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

        #probrvea_transformer = Normalizer().fit(int_probrvea.population.objectives)
        #norm_probrvea = probrvea_transformer.transform(int_probrvea.population.objectives)

        # R-metric calls for R_IGD and R_HV
        #rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_probrvea)
        #rigd_iprobrvea, rhv_iprobrvea = rmetric.calc(norm_probrvea, others=norm_rvea)

        #data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
        #    rigd_irvea,
        #    rhv_irvea,
        #]
        #data_row[["iProbRVEA" + excess_col for excess_col in excess_columns]] = [
        #    rigd_iprobrvea,
        #    rhv_iprobrvea,
        #]

        #data = data.append(data_row, ignore_index=1)

    # Decision phase
    # After the learning phase the reference vector which has the maximum number of assigned solutions forms ROI
    max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)

    for i in range(D):
        #data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
        #    problem_name,
        #    n_obj,
        #    L + i + 1,
        #    gen,
        #]

        # since composite front grows after each iteration this call should be done for each iteration
        base = baseADM(cf, reference_vectors)
        problem.ideal = np.asarray(base.ideal_point)

        for approach in approaches:
            dict_moea_objs[approach].population.ideal_objective_vector =  np.asarray(base.ideal_point)
            print("Ideal_"+approach+":", dict_moea_objs[approach].population.ideal_objective_vector)
        print("Problem Ideal:",problem.ideal)
        # generates the next reference point for the decision phase
        response = gp.generateRP4decision(base, max_assigned_vector[0])
        #data_row["reference_point"] = [
        #    response,
        #]
        print("Reference point:", response)
        dict_archive_all[i+L+1] = {}
        dict_archive_all[i+L+1]['reference_point'] = response
        dict_archive_all[i+L+1]['phase'] = 'D'
        for approach in approaches:
            # run algorithms with the new reference point
            _, pref_int_moea = dict_moea_objs[approach].requests()
            pref_int_moea.response = pd.DataFrame(
                [response], columns=pref_int_moea.content["dimensions_data"].columns
            )
            _, pref_int_moea = dict_moea_objs[approach].iterate(pref_int_moea)
            dict_archive_all[i+L+1][approach] = dict_moea_objs[approach].population

        # extend composite front with newly obtained solutions
        cf = generate_composite_front(
            dict_moea_objs[approaches[0]].population.objectives, 
            dict_moea_objs[approaches[1]].population.objectives, 
            do_nds=False
        )

        # R-metric calculation
        #ref_point = response.reshape(1, n_obj)

        #rp_transformer = Normalizer().fit(ref_point)
        #norm_rp = rp_transformer.transform(ref_point)

        # for decision phase, delta is specified as 0.2
        #rmetric = rm.RMetric(problemR, norm_rp, delta=0.2, pf=pareto_front)
        

        # normalize solutions before sending r-metric
        #rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
        #norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

        #probrvea_transformer = Normalizer().fit(int_probrvea.population.objectives)
        #norm_probrvea = probrvea_transformer.transform(int_probrvea.population.objectives)

        #rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_probrvea)
        #rigd_iprobrvea, rhv_iprobrvea = rmetric.calc(norm_probrvea, others=norm_rvea)
        #print("R HV generic:", rhv_irvea)
        #print("R HV prob:", rhv_iprobrvea)
        #data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
        #    rigd_irvea,
        #    rhv_irvea,
        #]
        #data_row[["iProbRVEA" + excess_col for excess_col in excess_columns]] = [
        #    rigd_iprobrvea,
        #    rhv_iprobrvea,
        #]

        #data = data.append(data_row, ignore_index=1)

#data.to_csv("./adm_emo/results/output_test_am.csv", index=False)
    print(dict_archive_all)
    return dict_archive_all