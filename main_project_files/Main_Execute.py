import sys
sys.path.insert(1, '/mnt/i/AmzNew/')
from desdeo_problem.Problem import DataProblem

from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from desdeo_emo.EAs.OfflineRVEA import ProbRVEAv3

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygmo import non_dominated_front_2d as nd2

from optproblems import dtlz
from pyDOE import lhs
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import scipy.io
import matplotlib.pyplot as plt
import pickle
import dill
from pymop.problems.welded_beam import WeldedBeam
from pymop.problems.truss2d import Truss2D
import warnings
from EI_calc import expected_improvement
warnings.filterwarnings("ignore")

np.random.seed(11)


def offline(problem_name, problem_testbench, folder_data, folder_runs, number_objectives,
            dimension_decisionvar, run_number, sampling, mode, emo_algorithm):

    print("Run Number:")
    print(run_number)
    class newProblem(baseProblem):
        """New problem description."""

        def __init__(
                self,
                name=None,
                num_of_variables=None,
                num_of_objectives=None,
                num_of_constraints=0,
                upper_limits=1,
                lower_limits=0,
                #upper_limits=np.array([5.0, 10.0, 10.0, 5.0]),
                #lower_limits=np.array([0.125, 0.1, 0.1, 0.125]),
                sample_size=109,
                mode=None,
                run=0,
                kf_cv=0
                # num_of_objectives_real

        ):

            super(newProblem, self).__init__(
                name,
                num_of_variables,
                num_of_objectives,
                num_of_constraints,
                upper_limits,
                lower_limits,
            )
            # self.num_of_objectives_real = None
            self.gp_list = []
            self.sample_size = sample_size
            self.initial_pop = []
            self.y_sample = []
            self.mode = mode
            self.run = run
            self.kf_cv = kf_cv
            self.err_all = None
            self.gp_err = []
            self.name = name
            self.r_squared_cv = None
            self.mean_r_squared_cv = None
            self.rmse_cv = None
            self.mean_rmse_cv = None
            if self.mode == 1:
                self.num_of_objectives_real = int(self.num_of_objectives)
            elif self.mode == 2:
                self.num_of_objectives_real = int(self.num_of_objectives / 2)
            elif self.mode == 3:
                self.num_of_objectives_real = int(self.num_of_objectives - 1)
            else:
                self.num_of_objectives_real = int(self.num_of_objectives)

            self.unc_marker = np.zeros(self.num_of_objectives_real)

            if sampling == "DIRECT":
                pass
            else:
                self.read_dataset()
                self.build_kriging()
                # Change the number of objectives depending on the CV
                print(self.mean_r_squared_cv)
                if self.mode == 4:
                    for i in range(self.num_of_objectives_real):
                        if self.mean_r_squared_cv[i, 0] < 0.9:
                            self.unc_marker[i] = 1
                            self.num_of_objectives += 1
                print(self.num_of_objectives)
                print(self.unc_marker)

        def f(self, x):
            """The function to predict."""
            if self.name == "DTLZ1":
                obj_val = dtlz.DTLZ1(self.num_of_objectives_real, self.num_of_variables)(x)

            elif self.name == "DTLZ2":
                obj_val = dtlz.DTLZ2(self.num_of_objectives_real, self.num_of_variables)(x)

            elif self.name == "DTLZ3":
                obj_val = dtlz.DTLZ3(self.num_of_objectives_real, self.num_of_variables)(x)

            elif self.name == "DTLZ4":
                obj_val = dtlz.DTLZ4(self.num_of_objectives_real, self.num_of_variables)(x)

            elif self.name == "DTLZ5":
                obj_val = dtlz.DTLZ5(self.num_of_objectives_real, self.num_of_variables)(x)

            elif self.name == "DTLZ6":
                obj_val = dtlz.DTLZ6(self.num_of_objectives_real, self.num_of_variables)(x)

            elif self.name == "DTLZ7":
                obj_val = dtlz.DTLZ7(self.num_of_objectives_real, self.num_of_variables)(x)

            elif self.name == "WELDED_BEAM":
                problem_weld = WeldedBeam()
                F, G = problem_weld.evaluate(x)
                obj_val = F

            elif self.name == "TRUSS2D":
                problem_truss = Truss2D()
                F, G = problem_truss.evaluate(x)
                obj_val = F

            return obj_val

        def read_dataset(self):

            mat = scipy.io.loadmat('./' + folder_data + '/Initial_Population_' + problem_testbench + '_' + sampling +
                                   '_AM_' + str(dimension_decisionvar) + '_109.mat')
            """
            
            if sampling == 'LHS':
                
                #mat = scipy.io.loadmat('./AM_Samples_109/Initial_Population_DTLZ_LHS_AM_new.mat')
                #mat = scipy.io.loadmat('./AM_Samples_109/Initial_Population_WELD_LHS_AM_109.mat')
                #mat = scipy.io.loadmat('./AM_Samples_109/Initial_Population_TRUSS2D_LHS_AM_109.mat')
                #mat = scipy.io.loadmat('./AM_Samples_109/Initial_Population_DDMOPP_LHS_AM_' +
                #                       str(self.num_of_variables) + '_109.mat')
            elif sampling == 'OPTRAND':
                mat = scipy.io.loadmat('./AM_Samples_109/Initial_Population_Optimal_Random_AM_109_new_'
                                       + self.name + 'Obj' + str(self.num_of_objectives_real) + '.mat')
            elif sampling == 'BETA':
                mat = scipy.io.loadmat('./AM_Samples_109/Initial_Population_DTLZ_BETA.mat')

            elif sampling == 'MVNORM':
                mat = scipy.io.loadmat('./AM_Samples_109/Initial_Population_DDMOPP_MVNORM_AM_' +
                                       str(self.num_of_variables) + '_109.mat')
                #mat = scipy.io.loadmat('./AM_Samples_109/Initial_Population_DTLZ_MVNORM3.mat')
            """
            #dataset = ((mat['Initial_Population_DTLZ'])[0][self.run])[0]
            #dataset = ((mat['Initial_Population_WELD'])[0][self.run])[0]
            #dataset = ((mat['Initial_Population_TRUSS2D'])[0][self.run])[0]
            #m = self.num_of_variables
            #n = 109
            #samples = lhs(m, samples=n)
            #dataset = samples
            dataset = ((mat['Initial_Population_'+problem_testbench])[0][self.run])[0]
            self.initial_pop = dataset
            #np.savetxt("data_weld.csv", dataset, delimiter=",")

        def build_kriging(self):
            x = self.initial_pop
            y = None
            if problem_testbench == 'DDMOPP':
                mat = scipy.io.loadmat('./'+folder_data+'/Obj_vals_DDMOPP_'+sampling+'_AM_'+self.name+'_'
                                       + str(self.num_of_objectives) + '_' + str(self.num_of_variables) + '_109.mat')
                y = ((mat['Obj_vals_DDMOPP'])[0][self.run])[0]
            else:
                for ind in x:
                    if y is None:
                        y = np.asarray(self.f(ind))
                    else:
                        y = np.vstack((y, self.f(ind)))
            self.y_sample = y
            self.gp_list = []
            # Instantiate a Gaussian Process model
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            kf = KFold(n_splits=10, random_state=None, shuffle=True)
            x_copy = x
            y_copy = y
            print("Building Kriging Models ...")
            # Fit to data using Maximum Likelihood Estimation of the parameters
            for i in range(self.num_of_objectives_real):
                # gp = GaussianProcessRegressor(alpha=1e-7, kernel=kernel, n_restarts_optimizer=9)
                r_sq = None
                rmse = None
                print("Building model", i + 1)
                count = 0
                mean_sd_test = None
                err_all = None
                if self.kf_cv == 1:
                    print("Building K models for CV ...")
                    for train_index, test_index in kf.split(x):
                        gp = GaussianProcessRegressor(alpha=0, kernel=kernel, n_restarts_optimizer=9)
                        gp.fit(x[train_index], y[train_index, i])
                        y_pred_test, sd_test = gp.predict(
                            x[test_index].reshape(np.size(x[test_index], axis=0), self.num_of_variables),
                            return_std=True)
                        count = count + 1
                        print(count)
                        # err = np.sqrt(np.square(y[test_index, i] - y_pred_test))
                        #err = y[test_index, i] - y_pred_test
                        #err_max = np.max(err)
                        # print("Error all = ", y[test_index, i] - y_pred_test)
                        # print("Error max = ", err_max)
                        ### Actually R sq is RMSE here
                        #print("R sq for fold", count)
                        #print(r2_score(y[test_index,i], y_pred_test))
                        #print(np.sqrt(mean_squared_error(y[test_index, i], y_pred_test)))
                        #print("Mean SD pred", np.mean(sd_test))
                        #if r_sq is None:
                        if r_sq is None:
                            r_sq = np.asarray(r2_score(y[test_index,i], y_pred_test))
                            rmse = np.asarray(np.sqrt(mean_squared_error(y[test_index, i], y_pred_test)))
                        else:
                            r_sq = np.hstack((r_sq, r2_score(y[test_index,i], y_pred_test)))
                            rmse = np.hstack((rmse,np.sqrt(mean_squared_error(y[test_index, i], y_pred_test))))
                        #mean_sd_test = np.asarray(np.mean(sd_test))
                        #err_all = np.asarray(err)
                        #self.mean_r_squared_cv = np.mean(self.r_squared_cv)
                    #else:
                        # r_sq = np.vstack((r_sq,np.asarray(r2_score(y[test_index,i], y_pred_test))))
                        #self.rmse_cv = np.vstack(
                        #    (r_sq, np.asarray(np.sqrt(mean_squared_error(y[test_index, i], y_pred_test)))))
                        #mean_sd_test = np.vstack((mean_sd_test, np.asarray(np.mean(sd_test))))
                        #self.mean_rmse_cv = np.mean(self.rmse_cv)
                        #err_all = np.hstack((err_all, err))
                        #print("R_squared =", r_sq)
                        #print("RMSE. =", rmse)
                    if self.mean_rmse_cv is None:
                        self.rmse_cv = rmse
                        self.r_squared_cv = r_sq
                        self.mean_rmse_cv = np.mean(rmse)
                        self.mean_r_squared_cv = np.mean(r_sq)

                    else:
                        self.rmse_cv = np.vstack((self.rmse_cv, rmse))
                        self.r_squared_cv = np.vstack((self.r_squared_cv,r_sq))
                        self.mean_rmse_cv = np.vstack((self.mean_rmse_cv,np.mean(rmse)))
                        self.mean_r_squared_cv = np.vstack((self.mean_r_squared_cv,np.mean(r_sq)))
                    print(self.r_squared_cv)
               # Building global model again
                gp = GaussianProcessRegressor(alpha=1e-7, kernel=kernel, n_restarts_optimizer=9)
                gp.fit(x_copy, y_copy[:, i])
                self.gp_list.append(gp)

            """if self.kf_cv == 1:
                self.err_all = np.transpose(self.err_all)
                print(self.err_all)
                for i in range(self.num_of_objectives_real):
                    gp = GaussianProcessRegressor(alpha=1e-7, kernel=kernel, n_restarts_optimizer=9)
                    gp.fit(x_copy, self.err_all[:, i])
                    self.gp_err.append(gp)
                    """
            print("Kriging Models build complete.")

        def objectives(self, decision_variables):
            if sampling == "DIRECT":
                y = self.f(decision_variables)
            else:
                y = None
                y_sigma = None
                sigma_bar = 0
                count = 0
                # for gp, gp_err in zip(self.gp_list, self.gp_err):
                for gp in self.gp_list:
                    count = count + 1
                    y_pred, sigma = gp.predict(decision_variables.reshape(1, self.num_of_variables), return_std=True)
                    # sigma = gp_err.predict(decision_variables.reshape(1, self.num_of_variables),
                    #                      return_std=False)
                    # sigma = np.absolute(sigma)
                    #if sigma == 0:
                    #    sigma = 0.1e-4
                    if self.mode == 1:
                        if y is None:
                            y = np.asarray(y_pred)
                        else:
                            y = np.hstack((y, y_pred))
                    elif self.mode == 2:
                        if y is None:
                            y = np.asarray(y_pred)
                            y_sigma = np.asarray(sigma)
                        else:
                            y = np.hstack((y, y_pred))
                            y_sigma = np.hstack((y_sigma, sigma))
                        if count == self.num_of_objectives_real:
                            y = np.hstack((y, y_sigma))
                    elif self.mode == 3:
                        sigma_bar = sigma_bar + sigma
                        # sigma_bar = sigma
                        # sigma_bar = y_pred_err
                        if y is None:
                            y = np.asarray(y_pred)

                        else:
                            y = np.hstack((y, y_pred))
                        if count == self.num_of_objectives_real:
                            y = np.hstack((y, sigma_bar / self.num_of_objectives_real))
                            # y = np.hstack((y, sigma_bar))
                    elif self.mode == 5:
                        ei = -expected_improvement(y_pred,sigma,self.y_sample[:, count-1])
                        if y is None:
                            y = np.asarray(ei)

                        else:
                            y = np.hstack((y, ei))
                    else:
                        if y is None:
                            y = np.asarray(y_pred)
                        else:
                            y = np.hstack((y, y_pred))

                        if self.unc_marker[count-1] == 1:
                            if y_sigma is None:
                                y_sigma = np.asarray(sigma)
                            else:
                                y_sigma = np.hstack((y_sigma, sigma))

                        if count == self.num_of_objectives_real and np.count_nonzero(self.unc_marker) > 0:
                            y = np.hstack((y, y_sigma))
            print(y)
            return y


    name = problem_name
    #k = 10
    if mode == 1:
        numobj = number_objectives
    elif mode == 2:
        numobj = number_objectives*2
    elif mode == 3:
        numobj = number_objectives + 1
    else:
        numobj = number_objectives
    numconst = 0
    #numvar = numobj + k - 1
    numvar = dimension_decisionvar

    kf_set=0
    if emo_algorithm == "MODEL_CV":
        kf_set = 1

    xu_weld = [5.0, 10.0, 10.0, 5.0]
    xl_weld = [0.125, 0.1, 0.1, 0.125]
    xl_truss = [0.0, 0.0, 1.0]
    xu_truss = [0.01, 1e5, 3.0]
    #problem = newProblem(name, numvar, numobj, numconst, mode=mode, kf_cv=0, run=run_number, upper_limits=xu_weld, lower_limits=xl_weld)
    #problem = newProblem(name, numvar, numobj, numconst, mode=mode, kf_cv=0, run=run_number, upper_limits=xu_truss,
    #                     lower_limits=xl_truss)
    if problem_testbench == "DDMOPP":
        lowerl=-1
    else:
        lowerl=0
    problem = newProblem(name, numvar, numobj, numconst, mode=mode, kf_cv=kf_set, run=run_number, lower_limits=lowerl)
    lattice_resolution = 4
    population_size = 105

    if sampling == "DIRECT":
        pop = Population(problem, plotting=False, run_number=run_number)
    else:
        pop = Population(problem, plotting=True, assign_type="init_samples", init_individuals=problem.initial_pop
                         , run_number=run_number, algorithm=emo_algorithm, mode=mode, folder_runs=folder_runs,
                         iter_plot=True)

    if emo_algorithm == "RVEA":
        pop.evolve(RVEA)
    elif emo_algorithm == "IBEA":
        pop.evolve(IBEA)
    elif emo_algorithm == "NSGAIII":
        pop.evolve(NSGAIII)
    else:
        pass

    if emo_algorithm == "MODEL_CV":
        results_dict = {
            'r_squared': problem.r_squared_cv,
            'mean_r_squared': problem.mean_r_squared_cv,
            'rmse': problem.rmse_cv,
            'mean_rmse': problem.mean_rmse_cv
        }
    else:
        results_dict = {
            'individual_archive': pop.individuals_archive,
            'objectives_archive': pop.objectives_archive,
            'obj_solutions': pop.objectives,
            'individuals_solutions': pop.individuals
        }

    print(results_dict)

    return results_dict
