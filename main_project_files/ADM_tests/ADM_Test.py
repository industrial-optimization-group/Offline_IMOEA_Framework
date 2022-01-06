import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/AmzNew/')

import numpy as np
import pandas as pd

import adm_emo.baseADM
from adm_emo.baseADM import *
import adm_emo.generatePreference as gp

from desdeo_emo.EAs.RVEA import RVEA
#from desdeo_emo.EAs.OfflineRVEA import ProbRVEAv3
from desdeo_emo.EAs.NSGAIII import NSGAIII as ProbRVEAv3

from desdeo_problem.Problem import DataProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor

from pymoo.factory import get_problem, get_reference_directions
import adm_emo.rmetric as rm
from sklearn.preprocessing import Normalizer
from pymoo.configuration import Configuration

from pyDOE import lhs
from sklearn.gaussian_process.kernels import Matern

Configuration.show_compile_hint = False

def compute_nadir(population):
    max_gen = None
    for i in population.objectives_archive:
        if max_gen is None:
            max_gen = np.amax(population.objectives_archive[i], axis=0)
        else:
            max_gen = np.amax(np.vstack((population.objectives_archive[i], max_gen)), axis=0)
    return max_gen

problem_names = ["DTLZ2"]

n_objs = np.asarray([3])  # number of objectives
K = 10
n_vars = K + n_objs - 1  # number of variables


num_gen_per_iter = [5]  # number of generations per iteration



algorithms = ["iRVEA", "iProbRVEA"]  # algorithms to be compared

# the followings are for formatting results
column_names = (
    ["problem", "num_obj", "iteration", "num_gens", "reference_point"]
    + [algorithm + "_R_IGD" for algorithm in algorithms]
    + [algorithm + "_R_HV" for algorithm in algorithms]
)
excess_columns = [
    "_R_IGD",
    "_R_HV",
]
data = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

# ADM parameters
L = 4  # number of iterations for the learning phase
D = 3  # number of iterations for the decision phase
lattice_resolution = 5  # density variable for creating reference vectors


counter = 1
total_count = len(num_gen_per_iter) * len(n_objs) * len(problem_names)
for gen in num_gen_per_iter:
    for n_obj, n_var in zip(n_objs, n_vars):
        for problem_name in problem_names:
            print(f"Loop {counter} of {total_count}")
            counter += 1
            #problem = test_problem_builder(
            #    name=problem_name, n_of_objectives=n_obj, n_of_variables=n_var
            #)
            prob = test_problem_builder(name=problem_name, n_of_objectives=n_obj, n_of_variables=n_var)
            x = lhs(n_var, 200)
            y = prob.evaluate(x)

            x_names = [f'x{i}' for i in range(1, n_var+1)]
            y_names = ["f1", "f2", "f3"]

            data = pd.DataFrame(np.hstack((x,y.objectives)), columns=x_names+y_names)
            problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names)
            problem.train(GaussianProcessRegressor, model_parameters={"kernel": Matern(nu=3/2)})
            
            #problem.ideal = np.asarray([0] * n_obj)
            #problem.nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 1

            #problem.ideal = np.asarray([-100] * n_obj)
            #problem.nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 100



            #true_nadir = np.asarray([1] * n_obj)

            # interactive
            int_rvea = RVEA(problem=problem, interact=True, use_surrogates=True, n_gen_per_iter=gen)
            int_probrvea = ProbRVEAv3(problem=problem, interact=True, use_surrogates=True, n_gen_per_iter=gen)

            # initial reference point is specified randomly
            response = np.random.rand(n_obj)

            # run algorithms once with the randomly generated reference point
            _, pref_int_rvea = int_rvea.requests()
            _, pref_int_probrvea = int_probrvea.requests()

            pref_int_rvea.response = pd.DataFrame(
                [response], columns=pref_int_rvea.content["dimensions_data"].columns
            )
            pref_int_probrvea.response = pd.DataFrame(
                [response], columns=pref_int_probrvea.content["dimensions_data"].columns
            )

            #int_rvea.population.ideal_objective_vector = np.asarray([-100] * n_obj)
            #int_probrvea.population.ideal_objective_vector = np.asarray([-100] * n_obj)

            _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
            _, pref_int_probrvea = int_probrvea.iterate(pref_int_probrvea)

            # build initial composite front
            cf = generate_composite_front(
                int_rvea.population.objectives, int_probrvea.population.objectives, do_nds=False
            )



            #base = baseADM(cf, reference_vectors)
            #problem.ideal = np.asarray(base.ideal_point)
            #int_rvea.population.ideal_objective_vector =  np.asarray(base.ideal_point)
            #int_probrvea.population.ideal_objective_vector = np.asarray(base.ideal_point)
            #_, pref_int_rvea = int_rvea.requests()
            #_, pref_int_probrvea = int_probrvea.requests()

            # the following two lines for getting pareto front by using pymoo framework
            problemR = get_problem(problem_name.lower(), n_var, n_obj)
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            pareto_front = problemR.pareto_front(ref_dirs)

            # creates uniformly distributed reference vectors
            reference_vectors = ReferenceVectors(lattice_resolution, n_obj)

            # learning phase
            for i in range(L):
                data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                    problem_name,
                    n_obj,
                    i + 1,
                    gen,
                ]

                # After this class call, solutions inside the composite front are assigned to reference vectors
                base = baseADM(cf, reference_vectors)
                problem.ideal = np.asarray(base.ideal_point)
                int_rvea.population.ideal_objective_vector =  np.asarray(base.ideal_point)
                int_probrvea.population.ideal_objective_vector = np.asarray(base.ideal_point)
                print("Problem Ideal:",problem.ideal)
                # generates the next reference point for the next iteration in the learning phase
                response = gp.generateRP4learning(base)

                data_row["reference_point"] = [
                    response,
                ]
                print("Ideal generic:", int_rvea.population.ideal_objective_vector)
                print("Ideal probabilistic:", int_probrvea.population.ideal_objective_vector)
                print("Reference point:", response)
                _, pref_int_rvea = int_rvea.requests()
                _, pref_int_probrvea = int_probrvea.requests()
                # run algorithms with the new reference point
                pref_int_rvea.response = pd.DataFrame(
                    [response], columns=pref_int_rvea.content["dimensions_data"].columns
                )
                pref_int_probrvea.response = pd.DataFrame(
                    [response], columns=pref_int_probrvea.content["dimensions_data"].columns
                )


                _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
                _, pref_int_probrvea = int_probrvea.iterate(pref_int_probrvea)

                # extend composite front with newly obtained solutions
                cf = generate_composite_front(
                    cf, int_rvea.population.objectives, int_probrvea.population.objectives, do_nds=False
                )

                # R-metric calculation
                ref_point = response.reshape(1, n_obj)

                # normalize reference point
                rp_transformer = Normalizer().fit(ref_point)
                norm_rp = rp_transformer.transform(ref_point)

                rmetric = rm.RMetric(problemR, norm_rp, pf=pareto_front)

                # normalize solutions before sending r-metric
                rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

                probrvea_transformer = Normalizer().fit(int_probrvea.population.objectives)
                norm_probrvea = probrvea_transformer.transform(int_probrvea.population.objectives)

                # R-metric calls for R_IGD and R_HV
                rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_probrvea)
                rigd_iprobrvea, rhv_iprobrvea = rmetric.calc(norm_probrvea, others=norm_rvea)

                data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
                    rigd_irvea,
                    rhv_irvea,
                ]
                data_row[["iProbRVEA" + excess_col for excess_col in excess_columns]] = [
                    rigd_iprobrvea,
                    rhv_iprobrvea,
                ]

                data = data.append(data_row, ignore_index=1)

            # Decision phase
            # After the learning phase the reference vector which has the maximum number of assigned solutions forms ROI
            max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)

            for i in range(D):
                data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                    problem_name,
                    n_obj,
                    L + i + 1,
                    gen,
                ]

                # since composite front grows after each iteration this call should be done for each iteration
                base = baseADM(cf, reference_vectors)

                problem.ideal = np.asarray(base.ideal_point)
                int_rvea.population.ideal_objective_vector =  np.asarray(base.ideal_point)
                int_probrvea.population.ideal_objective_vector = np.asarray(base.ideal_point)
                print("Problem Ideal:",problem.ideal)

                # generates the next reference point for the decision phase
                response = gp.generateRP4decision(base, max_assigned_vector[0])

                data_row["reference_point"] = [
                    response,
                ]
                print("Ideal generic:", int_rvea.population.ideal_objective_vector)
                print("Ideal probabilistic:", int_probrvea.population.ideal_objective_vector)
                print("Reference point:", response)
                _, pref_int_rvea = int_rvea.requests()
                _, pref_int_probrvea = int_probrvea.requests()
                
                # run algorithms with the new reference point
                pref_int_rvea.response = pd.DataFrame(
                    [response], columns=pref_int_rvea.content["dimensions_data"].columns
                )
                pref_int_probrvea.response = pd.DataFrame(
                    [response], columns=pref_int_probrvea.content["dimensions_data"].columns
                )

                _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
                _, pref_int_probrvea = int_probrvea.iterate(pref_int_probrvea)

                # extend composite front with newly obtained solutions
                cf = generate_composite_front(
                    cf, int_rvea.population.objectives, int_probrvea.population.objectives, do_nds=False
                )

                # R-metric calculation
                ref_point = response.reshape(1, n_obj)

                rp_transformer = Normalizer().fit(ref_point)
                norm_rp = rp_transformer.transform(ref_point)

                # for decision phase, delta is specified as 0.2
                rmetric = rm.RMetric(problemR, norm_rp, delta=0.2, pf=pareto_front)
                

                # normalize solutions before sending r-metric
                rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

                probrvea_transformer = Normalizer().fit(int_probrvea.population.objectives)
                norm_probrvea = probrvea_transformer.transform(int_probrvea.population.objectives)

                rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_probrvea)
                rigd_iprobrvea, rhv_iprobrvea = rmetric.calc(norm_probrvea, others=norm_rvea)
                print("R HV generic:", rhv_irvea)
                print("R HV prob:", rhv_iprobrvea)
                data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
                    rigd_irvea,
                    rhv_irvea,
                ]
                data_row[["iProbRVEA" + excess_col for excess_col in excess_columns]] = [
                    rigd_iprobrvea,
                    rhv_iprobrvea,
                ]

                data = data.append(data_row, ignore_index=1)

data.to_csv("./adm_emo/results/output_test_am.csv", index=False)