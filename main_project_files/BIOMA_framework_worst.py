from desdeo_problem.Problem import DataProblem

from desdeo_problem.surrogatemodels.SurrogateModels import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from desdeo_emo.EAs.OfflineRVEA import ProbRVEAv3
from desdeo_emo.EAs.OfflineRVEAnew import ProbRVEAv3
from desdeo_emo.EAs.OfflineRVEAnew import ProbRVEAv1
from desdeo_emo.EAs.OfflineRVEAnew import ProbRVEAv0_pump
from desdeo_emo.EAs.OfflineRVEAnew import ProbRVEAv1_pump
from desdeo_emo.EAs.OfflineRVEAnew import ProbRVEAv2_pump
from desdeo_emo.EAs.OfflineRVEA import RVEA

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from other_tools.non_domx import ndx
import plotting_tools.plot_interactive as plt_int2
import plotting_tools.plot3d_confidence as plt_int3
import plotting_tools.plot_reference_vectors as plt_refv


"""
def build_models(x, y):
    x_names = [f'x{i}' for i in range(1,31)]
    y_names = ["f1", "f2"]

    data = pd.DataFrame(np.hstack((x,y.objectives)), columns=x_names+y_names)
    #data_pareto = ndx(y.objectives)
    #y.objectives[data_pareto]
    problem = DataProblem(data=data, variable_names=x_names, objective_names=y_names)
    problem.train(GaussianProcessRegressor, {"kernel": Matern(nu=3/2)})
    return problem
"""

def interactive_optimize_test(problem, path):
    #evolver_opt = RVEA(problem, use_surrogates=True, interact=True, n_gen_per_iter=10)
    evolver_opt = ProbRVEAv1(problem, use_surrogates=True, interact=True, n_gen_per_iter=10)
    plot, pref = evolver_opt.requests()   
    pref_last = None 
    while evolver_opt.continue_evolution():
        print(pref.content['message'])
        if evolver_opt._iteration_counter>1:
            print("Enter preferences:")
            pref.response = pd.DataFrame([[5,5]], columns=pref.content['dimensions_data'].columns)
            pref_last= pref.response
            plot, pref = evolver_opt.iterate(pref)
        else:
            plot, pref = evolver_opt.iterate()
        # enter preferences

        if evolver_opt._iteration_counter>2:
            uncertainty_interaction(evolver_opt=evolver_opt, pref = pref_last.to_numpy().flatten(), path=path)
    return evolver_opt

def compute_nadir(population):
    max_gen = None
    for i in population.objectives_archive:
        if max_gen is None:
            max_gen = np.amax(population.objectives_archive[i], axis=0)
        else:
            max_gen = np.amax(np.vstack((population.objectives_archive[i], max_gen)), axis=0)
    return max_gen


def full_optimize(problem, classification_model, gen_per_iter, max_iter, FE_max, selection_type):
    if selection_type == 'prob_only':
        evolver_opt = ProbRVEAv0_pump(classification_model=classification_model, 
                                    problem=problem, 
                                    use_surrogates=True, 
                                    interact=False, 
                                    n_gen_per_iter=gen_per_iter, 
                                    n_iterations=max_iter,
                                    total_function_evaluations=FE_max)
    
    elif selection_type == 'prob_class_v1':
        evolver_opt = ProbRVEAv1_pump(classification_model=classification_model, 
                                    problem=problem, 
                                    use_surrogates=True, 
                                    interact=False, 
                                    n_gen_per_iter=gen_per_iter, 
                                    n_iterations=max_iter,
                                    total_function_evaluations=FE_max)
    elif selection_type == 'prob_class_v2':
        evolver_opt = ProbRVEAv2_pump(classification_model=classification_model, 
                                    problem=problem, 
                                    use_surrogates=True, 
                                    interact=False, 
                                    n_gen_per_iter=gen_per_iter, 
                                    n_iterations=max_iter,
                                    total_function_evaluations=FE_max)
    else:
        evolver_opt = RVEA(problem, 
                        use_surrogates=True,
                        interact=False, 
                        n_gen_per_iter=gen_per_iter, 
                        n_iterations=max_iter,
                        total_function_evaluations=FE_max)    

    while evolver_opt.continue_evolution():
        evolver_opt.iterate()
        print("Population size:",np.shape(evolver_opt.population.objectives)[0])
    return evolver_opt


def interactive_optimize(problem, classification_model, gen_per_iter, max_iter, path):
    evolver_opt = ProbRVEAv1_pump(classification_model=classification_model, 
                                problem=problem, 
                                use_surrogates=True, 
                                interact=True, 
                                n_gen_per_iter=gen_per_iter, 
                                n_iterations=max_iter)
    #evolver_opt = RVEA(problem, use_surrogates=True, interact=True, n_gen_per_iter=gen_per_iter, n_iterations=max_iter)
    plot, pref = evolver_opt.requests()   
    pref_last = None
    ideal = None
    ideal_prev = np.ones(problem.n_of_objectives)*-100
    nadir = None
    while evolver_opt.continue_evolution():
        print("Iteration Count:")
        print(evolver_opt._iteration_counter)        
        if evolver_opt._iteration_counter>=1:
            print(pref.content['message'])
            refpoint=np.zeros(problem.n_of_objectives)
            print("Enter preferences:")
            for index in range(problem.n_of_objectives):
                while True:
                    print("Preference for objective ", index + 1)
                    print("Ideal value = ", ideal[index])
                    print("Nadir value = ", nadir[index])
                    pref_val = float(
                        input("Please input a value between ideal and nadir: ")
                    )
                    #if pref_val > ideal[index] and pref_val < nadir[index]:
                    if pref_val > ideal[index]:
                        refpoint[index] = pref_val
                        #refpoint[index] = ideal[index] + 0.5
                        break
            ideal_prev = ideal
            refpoint = np.reshape(refpoint,(1,-1))
            print("Reference point=",refpoint)
            pref.response = pd.DataFrame(refpoint, columns=pref.content['dimensions_data'].columns)
            pref_last= pref.response
            plot, pref = evolver_opt.iterate(pref)        
        else:
            plot, pref = evolver_opt.iterate()        

        nadir = compute_nadir(evolver_opt.population)
        evolver_opt.population.nadir_fitness_val = nadir
        print("Nadir point:",nadir)
        ideal = evolver_opt.population.ideal_fitness_val

        if evolver_opt._iteration_counter<=1:
            pref_rv = np.ones(problem.n_of_objectives)
        else:
            pref_rv = pref_last.to_numpy().flatten()
        refpoint = pref_rv - ideal_prev
        norm = np.sqrt(np.sum(np.square(refpoint)))
        refpoint = refpoint / norm
        print("Normalized reference point=", refpoint)
        plt_refv.plot_refv(objs=evolver_opt.reference_vectors.values, 
                            preference=refpoint, 
                            iteration=evolver_opt._iteration_counter, 
                            ideal=np.zeros(problem.n_of_objectives), 
                            nadir=np.ones(problem.n_of_objectives),
                            path=path)
        
        # enter preferences
        if evolver_opt._iteration_counter>=2:
            objs_interation_end, unc_interaction_end = uncertainty_interaction(evolver_opt=evolver_opt, pref = pref_last.to_numpy().flatten(), path=path)
            evolver_opt.objs_interation_end = objs_interation_end
            evolver_opt.unc_interaction_end = unc_interaction_end
        else:
            evolver_opt.objs_interation_end = evolver_opt.population.objectives
            evolver_opt.unc_interaction_end = evolver_opt.population.uncertainity
    return evolver_opt

def uncertainty_interaction(evolver_opt, pref, path):
    max_range_x = evolver_opt.population.ideal_fitness_val[0]
    max_range_y = evolver_opt.population.ideal_fitness_val[1]
    min_range_x = evolver_opt.population.nadir_fitness_val[0]
    min_range_y = evolver_opt.population.nadir_fitness_val[1]
    obj_arch = None
    unc_arch = None
    indiv_arch = None
    obj_arch_all = None
    unc_arch_all = None
    use_all_archive = True
    start_gen = 1
    count_interaction_thresh = 0
    ref_pnt_normalized = None

    if use_all_archive is False:
        start_gen = evolver_opt._current_gen_count - evolver_opt.n_gen_per_iter
    last_gen_objs = evolver_opt.population.objectives
    last_gen_indiv = evolver_opt.population.individuals
    last_gen_unc = evolver_opt.population.uncertainity

    print("Number of solutions in last generation:")
    print(np.shape(evolver_opt.population.objectives)[0])
    for i in range(start_gen, evolver_opt.population.gen_count):
        if obj_arch is None:
            obj_arch = evolver_opt.population.objectives_archive[str(i)]
            unc_arch = evolver_opt.population.uncertainty_archive[str(i)]
            indiv_arch = evolver_opt.population.individuals_archive[str(i)]
        else:
            obj_arch = np.vstack((obj_arch, evolver_opt.population.objectives_archive[str(i)]))
            unc_arch = np.vstack((unc_arch, evolver_opt.population.uncertainty_archive[str(i)]))
            indiv_arch = np.vstack((indiv_arch, evolver_opt.population.individuals_archive[str(i)]))
    print("Number of solutions in archive:")
    print(np.shape(obj_arch)[0])
    obj_arch_all = obj_arch
    unc_arch_all = unc_arch
    indiv_arch_all = indiv_arch 
    # evolver_opt.reference_vectors.ref_point   
    if pref is not None:
        ideal = evolver_opt.population.ideal_fitness_val
        refpoint = pref
        refpoint = refpoint - ideal
        norm = np.sqrt(np.sum(np.square(refpoint)))
        ref_pnt_normalized = refpoint / norm
        print("Reference point:")
        print(refpoint)
        print("Reference point normalized:")
        print(ref_pnt_normalized)
        edge_adapted_vectors = evolver_opt.reference_vectors.get_adapted_egde_vectors(ref_pnt_normalized)
        print("The edge adapted vectors are:")
        print(edge_adapted_vectors)

        #refV = vectors.neighbouring_angles_current
        # Normalization - There may be problems here

        fmin = np.amin(obj_arch, axis=0)
        #fmin = self.params["fmin_iteration"]
        translated_fitness = obj_arch - fmin
        fitness_norm = np.linalg.norm(translated_fitness, axis=1)
        fitness_norm = np.repeat(fitness_norm, len(translated_fitness[0, :])).reshape(
            len(translated_fitness), len(translated_fitness[0, :])
        )
        normalized_fitness = np.divide(translated_fitness, fitness_norm)  # Checked, works.
        cosine = np.dot(normalized_fitness, np.transpose(edge_adapted_vectors))
        cosine_vectors = np.dot(edge_adapted_vectors, np.transpose(edge_adapted_vectors))
        if cosine[np.where(cosine > 1)].size:
            print(
                "RVEA.py line 60 cosine larger than 1 decreased to 1:"
            )
            cosine[np.where(cosine > 1)] = 1
        if cosine[np.where(cosine < 0)].size:
            print(
                "RVEA.py line 64 cosine smaller than 0 decreased to 0:"
            )
            cosine[np.where(cosine < 0)] = 0

        if cosine_vectors[np.where(cosine_vectors > 1)].size:
            print(
                "RVEA.py line 60 cosine larger than 1 decreased to 1:"
            )
            cosine_vectors[np.where(cosine_vectors > 1)] = 1
        if cosine_vectors[np.where(cosine_vectors < 0)].size:
            print(
                "RVEA.py line 64 cosine smaller than 0 decreased to 0:"
            )
            cosine_vectors[np.where(cosine_vectors < 0)] = 0

        # Calculation of angles between reference vectors and solutions
        theta = np.arccos(cosine)
        #theta_vectors = np.arccos(cosine_vectors)[
        #    np.triu_indices(np.shape(obj_arch)[1], np.shape(obj_arch)[1]-1)]
        theta_vectors_one = np.max(np.arccos(cosine_vectors), axis=1)
        print("Angle between edge vectors:", theta_vectors_one)
        print("Angles between solutions:", theta)
        #theta_vectors_one = np.ones(np.shape(obj_arch)[1])*theta_vectors[0]
        index_pref = []
        for i in range(np.shape(obj_arch)[0]):
            if(all(theta[i] <= theta_vectors_one)):
                index_pref.append(i)
        obj_arch = obj_arch[index_pref]
        unc_arch = unc_arch[index_pref]
        indiv_arch = indiv_arch[index_pref]
    obj_arch_pref = obj_arch
    unc_arch_pref = unc_arch
    indiv_arch_pref = indiv_arch
    print("Number of solutions in archive within preference:")
    print(np.shape(obj_arch)[0])

    # Non-dominated sort in generic approach (without uncertainty)
    if np.shape(obj_arch)[0]>0:
        # Non-dominated sort
        obj_unc_arch2 = obj_arch
        non_dom_front2 = ndx(obj_unc_arch2)
        obj_arch2= obj_arch[non_dom_front2[0][0]]
        unc_arch2 = unc_arch[non_dom_front2[0][0]]
        indiv_arch2 = indiv_arch[non_dom_front2[0][0]]
        obj_arch_nds2 = obj_arch2
        unc_arch_nds2 = unc_arch2
        indiv_arch_nds2 = indiv_arch2
        print("Number of solutions after non-dom sort (without uncertainty):")
        print(np.shape(obj_arch2)[0])
    else:
        obj_arch2 = evolver_opt.population.objectives_archive[str(evolver_opt.population.gen_count-1)]
        unc_arch2 = evolver_opt.population.uncertainty_archive[str(evolver_opt.population.gen_count-1)]
        indiv_arch2 = evolver_opt.population.individuals_archive[str(evolver_opt.population.gen_count - 1)]
        obj_arch_nds2 = obj_arch2
        unc_arch_nds2 = unc_arch2
        indiv_arch_nds2 = indiv_arch2

    # Non-dominated sort
    if np.shape(obj_arch)[0]>0:
        # Non-dominated sort
        obj_unc_arch = np.hstack((obj_arch, unc_arch))
        non_dom_front = ndx(obj_unc_arch)
        obj_arch = obj_arch[non_dom_front[0][0]]
        unc_arch = unc_arch[non_dom_front[0][0]]
        indiv_arch = indiv_arch[non_dom_front[0][0]]
        obj_arch_nds = obj_arch
        unc_arch_nds = unc_arch
        indiv_arch_nds = indiv_arch
        print("Number of solutions after non-dom sort (including uncertinty):")
        print(np.shape(obj_arch)[0])
    else:
        obj_arch = evolver_opt.population.objectives_archive[str(evolver_opt.population.gen_count-1)]
        unc_arch = evolver_opt.population.uncertainty_archive[str(evolver_opt.population.gen_count-1)]
        indiv_arch = evolver_opt.population.individuals_archive[str(evolver_opt.population.gen_count - 1)]
        obj_arch_nds = obj_arch
        unc_arch_nds = unc_arch
        indiv_arch_nds = indiv_arch


    # Select solutions within the preference
    np.savetxt(path+'/Obj_arch_all' + '_' +str(evolver_opt._iteration_counter)+'.csv', obj_arch_all, delimiter=",")
    np.savetxt(path+'/Unc_arch_all' + '_' + str(evolver_opt._iteration_counter) + '.csv', unc_arch_all, delimiter=",")
    np.savetxt(path+'/Indiv_arch_all' + '_' + str(evolver_opt._iteration_counter) + '.csv', indiv_arch_all,delimiter=",")
    np.savetxt(path+'/Obj_arch_nds' + '_' + str(evolver_opt._iteration_counter) + '.csv', obj_arch_nds, delimiter=",")
    np.savetxt(path+'/Unc_arch_nds' + '_' + str(evolver_opt._iteration_counter) + '.csv', unc_arch_nds, delimiter=",")
    np.savetxt(path+'/Indiv_arch_nds' + '_' + str(evolver_opt._iteration_counter) + '.csv', indiv_arch_nds,
                delimiter=",")
    np.savetxt(path+'/Obj_arch_nds2' + '_' + str(evolver_opt._iteration_counter) + '.csv', obj_arch_nds2, delimiter=",")
    np.savetxt(path+'/Unc_arch_nds2' + '_' + str(evolver_opt._iteration_counter) + '.csv', unc_arch_nds2, delimiter=",")
    np.savetxt(path+'/Indiv_arch_nds2' + '_' + str(evolver_opt._iteration_counter) + '.csv', indiv_arch_nds2,
                delimiter=",")
    np.savetxt(path+'/Obj_arch_pref' + '_' + str(evolver_opt._iteration_counter) + '.csv', obj_arch_pref, delimiter=",")
    np.savetxt(path+'/Unc_arch_pref' + '_' + str(evolver_opt._iteration_counter) + '.csv', unc_arch_pref, delimiter=",")
    np.savetxt(path+'/Indiv_arch_pref' + '_' + str(evolver_opt._iteration_counter) + '.csv', indiv_arch_pref,
                delimiter=",")
    
    np.savetxt(path+'/Obj_arch_pref_prob' + '_' + str(evolver_opt._iteration_counter) + '.csv', last_gen_objs, delimiter=",")
    np.savetxt(path+'/Unc_arch_pref_prob' + '_' + str(evolver_opt._iteration_counter) + '.csv', last_gen_unc, delimiter=",")
    np.savetxt(path+'/Indiv_arch_pref_prob' + '_' + str(evolver_opt._iteration_counter) + '.csv', last_gen_indiv,
                delimiter=",")

    # Convert uncertainty to indifference
    unc_arch_percent = np.abs((1.96*unc_arch/obj_arch)*100)
    unc_arch_lower = obj_arch - 1.96*unc_arch
    unc_arch_upper = obj_arch + 1.96*unc_arch

    upper_obj_vals = obj_arch + 1.96*unc_arch
    min_upper = np.min(upper_obj_vals, axis=0)
    max_upper = np.max(upper_obj_vals, axis=0)

    min_uncertainty = np.min(unc_arch_percent, axis=0)
    max_uncertainty = np.max(unc_arch_percent, axis=0)
    thresholds = np.ones_like(evolver_opt.population.ideal_fitness_val)*np.inf

    # Plotting
    if np.shape(obj_arch)[1] == 2:
        if evolver_opt._iteration_counter==1:
            fig = plt.figure(1, figsize=(6, 6))
            ax = fig.add_subplot(111)
            ax.set_xlabel('$f_1$')
            ax.set_ylabel('$f_2$')
            plt.xlim(min_range_x, max_range_x)
            plt.ylim(min_range_y, max_range_y)
            unc_avg_all = np.mean(unc_arch_all, axis=1)
            unc_avg_all_max = np.max(unc_avg_all)
            unc_avg_all_min = np.min(unc_avg_all)
            unc_avg_all = unc_avg_all / unc_avg_all_max

            # ax.scatter(obj_arch_all[:, 0], obj_arch_all[:, 1], c=unc_avg)
            ax.scatter(obj_arch_all[:, 0], obj_arch_all[:, 1], c=unc_avg_all, vmax=1, vmin=unc_avg_all_min)
            plt.tight_layout()
            fig.savefig(path+'/threshold_' + str(0)
                        + '_' + str(0) + '.pdf')

        fig = plt.figure(1, figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        plt.xlim(min_range_x, max_range_x)
        plt.ylim(min_range_y, max_range_y)
        unc_avg_all = np.mean(unc_arch_all, axis=1)
        unc_avg_all_max = np.max(unc_avg_all)
        unc_avg_all_min = np.min(unc_avg_all)
        unc_avg_all = (unc_avg_all - unc_avg_all_min) / (unc_avg_all_max - unc_avg_all_min)
        unc_avg_pref = np.mean(unc_arch_pref, axis=1)
        unc_avg_pref = (unc_avg_pref - unc_avg_all_min)/ (unc_avg_all_max - unc_avg_all_min)
        unc_avg_nds = np.mean(unc_arch_nds, axis=1)
        unc_avg_nds = (unc_avg_nds - unc_avg_all_min)/ (unc_avg_all_max - unc_avg_all_min)
        #ax.scatter(obj_arch_all[:, 0], obj_arch_all[:, 1], c=unc_avg)
        #ax.scatter(obj_arch_all[:, 0], obj_arch_all[:, 1], c=unc_avg_all, alpha=0.3, vmax=1, vmin=unc_avg_all_min)

        ax.scatter(obj_arch_all[:, 0], obj_arch_all[:, 1], c=unc_avg_all, vmax=1, vmin=unc_avg_all_min)
        # ax.errorbar(obj_arch[:, 0], obj_arch[:, 1], xerr=1.96 * unc_arch[:, 0],
        #            yerr=1.96 * unc_arch[:, 1],
        #            fmt='o', ecolor='g')
        #if pref is not None:
        #    ax.scatter(pref[0], pref[1], c='r')
        # plt.show()
        plt.tight_layout()
        fig.savefig(path+'/all_' + str(evolver_opt._iteration_counter)
                    + '_' + str(0) + '.pdf')

        plt.clf()
        fig = plt.figure(1, figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        plt.xlim(min_range_x, max_range_x)
        plt.ylim(min_range_y, max_range_y)

        ax.scatter(obj_arch_all[:, 0], obj_arch_all[:, 1], c='lightgray')
        ax.scatter(obj_arch_pref[:, 0], obj_arch_pref[:, 1], c=unc_avg_pref, vmax=1, vmin=unc_avg_all_min)
        #ax.errorbar(obj_arch[:, 0], obj_arch[:, 1], xerr=1.96 * unc_arch[:, 0],
        #            yerr=1.96 * unc_arch[:, 1],
        #            fmt='o', ecolor='g')
        if pref is not None:
            ax.scatter(pref[0], pref[1], c='r',s=70)
        #plt.show()
        plt.tight_layout()
        fig.savefig(path+'/pref_' + str(evolver_opt._iteration_counter)
                    + '_' + str(0) + '.pdf')
        plt.clf()
        fig = plt.figure(1, figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        plt.xlim(min_range_x, max_range_x)
        plt.ylim(min_range_y, max_range_y)

        #fig.savefig('x1_start.pdf')
        ax.scatter(obj_arch_all[:, 0], obj_arch_all[:, 1], c='lightgray')
        ax.scatter(obj_arch_nds[:, 0], obj_arch_nds[:, 1], c=unc_avg_nds, vmax=1, vmin=unc_avg_all_min)
        #ax.errorbar(obj_arch[:, 0], obj_arch[:, 1], xerr=1.96 * unc_arch[:, 0],
        #            yerr=1.96 * unc_arch[:, 1],
        #            fmt='o', ecolor='g')
        if pref is not None:
            ax.scatter(pref[0], pref[1], c='r',s=70)
        #plt.show()
        plt.tight_layout()
        fig.savefig(path+'/nds_' + str(evolver_opt._iteration_counter)
                    + '_' + str(0) + '.pdf')
        plt.clf()
        print('Plotted!')
    elif np.shape(obj_arch)[1] > 2:
        unc_avg_all = np.mean(unc_arch_all, axis=1)
        unc_avg_all_max = np.max(unc_avg_all)
        unc_avg_all_min = np.min(unc_avg_all)

        plt_int3.plot_vals(objs=last_gen_objs,
                unc=last_gen_unc,
                preference=pref,
                iteration=evolver_opt._iteration_counter,
                interaction_count=-7,
                min=unc_avg_all_min,
                max=unc_avg_all_max,
                ideal=evolver_opt.population.ideal_fitness_val,
                nadir=evolver_opt.population.nadir_fitness_val,path=path)

        plt_int2.plot_vals(objs=obj_arch2,
                    unc=unc_arch2,
                    preference=pref,
                    iteration=evolver_opt._iteration_counter,
                    interaction_count=-3,
                    min=unc_avg_all_min,
                    max=unc_avg_all_max,
                    ideal=evolver_opt.population.ideal_fitness_val,
                    nadir=evolver_opt.population.nadir_fitness_val,path=path)

        plt_int2.plot_vals(objs=obj_arch_all,
                            unc=unc_arch_all,
                            preference=pref,
                            iteration=evolver_opt._iteration_counter,
                            min=unc_avg_all_min,
                            max=unc_avg_all_max,
                            interaction_count=-2,
                            ideal=evolver_opt.population.ideal_fitness_val,
                            nadir=evolver_opt.population.nadir_fitness_val,path=path)

        plt_int2.plot_vals(objs=obj_arch_pref,
                            unc=unc_arch_pref,
                            preference=pref,
                            iteration=evolver_opt._iteration_counter,
                            interaction_count=-1,
                            min=unc_avg_all_min,
                            max=unc_avg_all_max,
                            ideal=evolver_opt.population.ideal_fitness_val,
                            nadir=evolver_opt.population.nadir_fitness_val,path=path)

        plt_int3.plot_vals(objs=obj_arch_pref,
                            unc=unc_arch_pref,
                            preference=pref,
                            iteration=evolver_opt._iteration_counter,
                            interaction_count=-1,
                            min=unc_avg_all_min,
                            max=unc_avg_all_max,
                            ideal=evolver_opt.population.ideal_fitness_val,
                            nadir=evolver_opt.population.nadir_fitness_val,path=path)


        plt_int2.plot_vals(objs=obj_arch,
                            unc=unc_arch,
                            preference=pref,
                            iteration=evolver_opt._iteration_counter,
                            interaction_count=0,
                            min=unc_avg_all_min,
                            max=unc_avg_all_max,
                            ideal=evolver_opt.population.ideal_fitness_val,
                            nadir=evolver_opt.population.nadir_fitness_val,path=path)
        
        plt_int3.plot_vals(objs=obj_arch,
                    unc=unc_arch,
                    preference=pref,
                    iteration=evolver_opt._iteration_counter,
                    interaction_count=0,
                    min=unc_avg_all_min,
                    max=unc_avg_all_max,
                    ideal=evolver_opt.population.ideal_fitness_val,
                    nadir=evolver_opt.population.nadir_fitness_val,path=path)


        


    print("Total solutions after pre-filtering:")
    print(np.shape(obj_arch)[0])
    end_disp = 1.0
    end_disp = float(input("Do you want to SET upper threshold of ojective values? : "))

    while end_disp > 0.0:
        no_solns = True
        while no_solns is True:
            for index in range(len(thresholds)):
                while True:
                    print("Set the upper threshold for objective (in %)", index + 1)
                    print("Minimum upper value % = ", min_upper[index])
                    print("Maximum upper value % = ", max_upper[index])
                    thresh_val = input("Please input the upper threshold :  ")
                    print(thresh_val)
                    thresh_val = float(thresh_val)
                    print(thresh_val)
                    if thresh_val < max_upper[index]:
                        thresholds[index] = thresh_val
                        break
            loc = np.where(np.all(np.tile(thresholds,(np.shape(upper_obj_vals)[0],1))> upper_obj_vals, axis=1))
            if np.size(loc)>0:
                no_solns = False
            else:
                print("No solutions! Please re-enter preferences.")
        count_interaction_thresh += 1

        # Saving data
        np.savetxt(path+'/Obj_arch_cutoff' + '_' + str(evolver_opt._iteration_counter)
                    + '_' + str(count_interaction_thresh) + '.csv', obj_arch[loc],
                    delimiter=",")
        np.savetxt(path+'/Unc_arch_cutoff' + '_' + str(evolver_opt._iteration_counter)
                    + '_' + str(count_interaction_thresh) + '.csv', unc_arch[loc],
                    delimiter=",")
        np.savetxt(path+'/Unc_percent_arch_cutoff' + '_' + str(evolver_opt._iteration_counter)
                    + '_' + str(count_interaction_thresh) + '.csv', unc_arch_percent[loc],
                    delimiter=",")
        np.savetxt(path+'/Unc_lower_arch_cutoff' + '_' + str(evolver_opt._iteration_counter)
                    + '_' + str(count_interaction_thresh) + '.csv', unc_arch_lower[loc],
                    delimiter=",")
        np.savetxt(path+'/Unc_upper_arch_cutoff' + '_' + str(evolver_opt._iteration_counter)
                    + '_' + str(count_interaction_thresh) + '.csv', unc_arch_upper[loc],
                    delimiter=",")
        np.savetxt(path+'/Indiv_arch_cutoff' + '_' + str(evolver_opt._iteration_counter)
                    + '_' + str(count_interaction_thresh) + '.csv', indiv_arch[loc],
                    delimiter=",")
        # Plotting
        if np.shape(obj_arch)[1] == 2:
            fig = plt.figure(1, figsize=(6, 6))
            ax = fig.add_subplot(111)
            ax.set_xlabel('$f_1$')
            ax.set_ylabel('$f_2$')
            plt.xlim(min_range_x, max_range_x)
            plt.ylim(min_range_y, max_range_y)
            unc_avg = np.mean(unc_arch, axis=1)
            unc_avg = (unc_avg - unc_avg_all_min)/ (unc_avg_all_max - unc_avg_all_min)
            #unc_avg = unc_avg / unc_avg_all_max
            #ax.scatter(obj_arch_all[:, 0], obj_arch_all[:, 1], c=unc_avg_all, alpha=0.3, vmax=1,
            #           vmin=unc_avg_all_min)
            ax.scatter(obj_arch_all[:, 0], obj_arch_all[:, 1], c='gray')
            #ax.errorbar(obj_arch[loc, 0], obj_arch[loc, 1], xerr=1.96 * unc_arch[loc, 0],
            #             yerr=1.96 * unc_arch[loc, 1],
            #             fmt='o', ecolor='g')
            ax.scatter(obj_arch[loc, 0], obj_arch[loc, 1], c=np.reshape((unc_avg[loc]),(1,-1)), vmax=1, vmin=unc_avg_all_min)
            if pref is not None:
                ax.scatter(pref[0], pref[1], c='r',s=70)
            #plt.show()
            plt.tight_layout()
            fig.savefig(path+'/threshold_'+ str(evolver_opt._iteration_counter)
                        + '_' + str(count_interaction_thresh) + '.pdf')
            print('Plotted!')
        elif np.shape(obj_arch)[1] > 2:
            plt_int2.plot_vals(objs=obj_arch[loc],
                                unc=unc_arch[loc],
                                preference=pref,
                                iteration=evolver_opt._iteration_counter,
                                interaction_count=count_interaction_thresh,
                                min=unc_avg_all_min,
                                max=unc_avg_all_max,
                                ideal=evolver_opt.population.ideal_fitness_val,
                                nadir=evolver_opt.population.nadir_fitness_val,
                                path=path)
            plt_int3.plot_vals(objs=obj_arch[loc],
                    unc=unc_arch[loc],
                    preference=pref,
                    iteration=evolver_opt._iteration_counter,
                    interaction_count=count_interaction_thresh,
                    min=unc_avg_all_min,
                    max=unc_avg_all_max,
                    ideal=evolver_opt.population.ideal_fitness_val,
                    nadir=evolver_opt.population.nadir_fitness_val,
                    path=path)

        end_disp = float(input("Do you want to reset thresholds : "))
        print(end_disp)
    if end_disp > 0.0: 
        return obj_arch[loc], unc_arch[loc]
    else:
        return obj_arch, unc_arch

