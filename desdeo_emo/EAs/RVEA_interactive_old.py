from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
from pyRVEA.Selection.APD_select_interactive import APD_select
from pyRVEA.EAs.baseEA import BaseDecompositionEA
from pyRVEA.OtherTools.ReferenceVectors import ReferenceVectors
from non_domx import ndx
import numpy as np
from non_domx import ndx
import plot_interactive as plt_int
from pyRVEA.OtherTools.plotlyanimate import animate_init_, animate_next_

plt.rcParams["text.usetex"] = True
plt.rcParams.update({'font.size': 22})

if TYPE_CHECKING:
    from pyRVEA.Population.Population_interactive import Population


class RVEA(BaseDecompositionEA):
    """The python version reference vector guided evolutionary algorithm.

    See the details of RVEA in the following paper

    R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, A Reference Vector Guided
    Evolutionary Algorithm for Many-objective Optimization, IEEE Transactions on
    Evolutionary Computation, 2016

    The source code of pyRVEA is implemented by Bhupinder Saini

    If you have any questions about the code, please contact:

    Bhupinder Saini: bhupinder.s.saini@jyu.fi

    Project researcher at University of Jyväskylä.
    """

    def set_params(
        self,
        population: "Population" = None,
        population_size: int = None,
        lattice_resolution: int = None,
        interact: bool = True,
        a_priori_preference: bool = False,
        generations_per_iteration: int = 100,
        iterations: int = 100,
        Alpha: float = 2,
        plotting: bool = True,
        algorithm_name="RVEA",
    ):
        """Set up the parameters. Save in RVEA.params. Note, this should be
        changed to align with the current structure.

        Parameters
        ----------
        population : Population
            Population object
        population_size : int
            Population Size
        lattice_resolution : int
            Lattice resolution
        interact : bool
            bool to enable or disable interaction. Enabled if True
        a_priori_preference : bool
            similar to interact
        generations_per_iteration : int
            Number of generations per iteration.
        iterations : int
            Total Number of iterations.
        Alpha : float
            The alpha parameter of APD selection.
        plotting : bool
            Useless really.
        Returns
        -------

        """

        lattice_resolution_options = {
            "2": 49,
            "3": 13,
            "4": 7,
            "5": 5,
            "6": 4,
            "7": 3,
            "8": 3,
            "9": 3,
            "10": 3,
        }

        if population.problem.num_of_objectives < 11:
            lattice_resolution = lattice_resolution_options[
                str(population.problem.num_of_objectives)
            ]
        else:
            lattice_resolution = 3
        rveaparams = {
            "population_size": population_size,
            "lattice_resolution": lattice_resolution,
            "interact": interact,
            "a_priori": a_priori_preference,
            "generations": generations_per_iteration,
            "maximum_func_evals": 40000,
            "iterations": iterations,
            "Alpha": Alpha,
            "ploton": plotting,
            "current_iteration_gen_count": 0,
            "current_iteration_count": 0,
            "ref_pnt_disp": None,
            "fmin_iteration": None,
            "ref_pnt_normalized": None,
            "reference_vectors": ReferenceVectors(
                lattice_resolution, population.problem.num_of_objectives
            ),
        }
        return rveaparams

    def _plot_solutions(self,population:"Population"):
        fig = plt.figure(1, figsize=(6, 6))
        ax = fig.add_subplot(111)
        min_fitness = np.amin(population.fitness,axis=0)
        max_fitness = np.amax(population.fitness, axis=0)
        plt.xlim(min_fitness[0]-0.5, max_fitness[0]+0.5)
        plt.ylim(min_fitness[1]-0.5, max_fitness[1]+0.5)
        plt.errorbar(population.fitness[:, 0], population.fitness[:, 1], xerr=1.96 * population.fitness[:, 2],
                     yerr=1.96 * population.fitness[:, 3],
                     fmt='o', ecolor='g')
        #plt.scatter(vectors.values[:, 0], vectors.values[:, 1])
        #fig.savefig('interactive_test.pdf')
        plt.show()

    def plot_solutions2(self,population:"Population"):
        max_range_x = 0.5
        max_range_y = 0.7
        min_range_x = -0.1
        min_range_y = -0.2
        obj_arch = None
        unc_arch = None
        indiv_arch = None
        obj_arch_all = None
        unc_arch_all = None
        use_all_archive = True
        start_gen = 1
        count_interaction_thresh = 0

        if use_all_archive is False:
            start_gen = population.gen_count - self.params["generations"]
        print("Number of solutions in last generation:")
        print(np.shape(population.objectives)[0])
        for i in range(start_gen, population.gen_count):
            if obj_arch is None:
                obj_arch = population.objectives_archive[str(i)]
                unc_arch = population.uncertainty_archive[str(i)]
                indiv_arch = population.individuals_archive[str(i)]
            else:
                obj_arch = np.vstack((obj_arch, population.objectives_archive[str(i)]))
                unc_arch = np.vstack((unc_arch, population.uncertainty_archive[str(i)]))
                indiv_arch = np.vstack((indiv_arch, population.individuals_archive[str(i)]))
        print("Number of solutions in archive:")
        print(np.shape(obj_arch)[0])
        obj_arch_all = obj_arch
        unc_arch_all = unc_arch
        indiv_arch_all = indiv_arch
        if self.params["ref_pnt_disp"] is not None:
            edge_adapted_vectors = self.params["reference_vectors"]\
                .get_adapted_egde_vectors(self.params["ref_pnt_normalized"])
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
            print("Number of solutions after non-dom sort:")
            print(np.shape(obj_arch2)[0])
        else:
            obj_arch2 = population.objectives_archive[str(population.gen_count-1)]
            unc_arch2 = population.uncertainty_archive[str(population.gen_count-1)]
            indiv_arch2 = population.individuals_archive[str(population.gen_count - 1)]
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
            print("Number of solutions after non-dom sort:")
            print(np.shape(obj_arch)[0])
        else:
            obj_arch = population.objectives_archive[str(population.gen_count-1)]
            unc_arch = population.uncertainty_archive[str(population.gen_count-1)]
            indiv_arch = population.individuals_archive[str(population.gen_count - 1)]
            obj_arch_nds = obj_arch
            unc_arch_nds = unc_arch
            indiv_arch_nds = indiv_arch


        # Select solutions within the preference
        np.savetxt('Obj_arch_all' + '_' +str(self.params["current_iteration_count"])+'.csv', obj_arch_all, delimiter=",")
        np.savetxt('Unc_arch_all' + '_' + str(self.params["current_iteration_count"]) + '.csv', unc_arch_all, delimiter=",")
        np.savetxt('Indiv_arch_all' + '_' + str(self.params["current_iteration_count"]) + '.csv', indiv_arch_all,delimiter=",")
        np.savetxt('Obj_arch_nds' + '_' + str(self.params["current_iteration_count"]) + '.csv', obj_arch_nds, delimiter=",")
        np.savetxt('Unc_arch_nds' + '_' + str(self.params["current_iteration_count"]) + '.csv', unc_arch_nds, delimiter=",")
        np.savetxt('Indiv_arch_nds' + '_' + str(self.params["current_iteration_count"]) + '.csv', indiv_arch_nds,
                   delimiter=",")
        np.savetxt('Obj_arch_nds2' + '_' + str(self.params["current_iteration_count"]) + '.csv', obj_arch_nds2, delimiter=",")
        np.savetxt('Unc_arch_nds2' + '_' + str(self.params["current_iteration_count"]) + '.csv', unc_arch_nds2, delimiter=",")
        np.savetxt('Indiv_arch_nds2' + '_' + str(self.params["current_iteration_count"]) + '.csv', indiv_arch_nds2,
                   delimiter=",")
        np.savetxt('Obj_arch_pref' + '_' + str(self.params["current_iteration_count"]) + '.csv', obj_arch_pref, delimiter=",")
        np.savetxt('Unc_arch_pref' + '_' + str(self.params["current_iteration_count"]) + '.csv', unc_arch_pref, delimiter=",")
        np.savetxt('Indiv_arch_pref' + '_' + str(self.params["current_iteration_count"]) + '.csv', indiv_arch_pref,
                   delimiter=",")

        # Convert uncertainty to indifference
        unc_arch_percent = np.abs((1.96*unc_arch/obj_arch)*100)
        min_uncertainty = np.min(unc_arch_percent, axis=0)
        max_uncertainty = np.max(unc_arch_percent, axis=0)
        thresholds = np.ones_like(population.ideal_fitness)*np.inf

        # Plotting
        if np.shape(obj_arch)[1] == 2:
            if self.params["current_iteration_count"]==1:
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
                fig.savefig('threshold_' + str(0)
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
            #if self.params["ref_pnt_disp"] is not None:
            #    ax.scatter(self.params["ref_pnt_disp"][0], self.params["ref_pnt_disp"][1], c='r')
            # plt.show()
            plt.tight_layout()
            fig.savefig('all_' + str(self.params["current_iteration_count"])
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
            if self.params["ref_pnt_disp"] is not None:
                ax.scatter(self.params["ref_pnt_disp"][0], self.params["ref_pnt_disp"][1], c='r',s=70)
            #plt.show()
            plt.tight_layout()
            fig.savefig('pref_' + str(self.params["current_iteration_count"])
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
            if self.params["ref_pnt_disp"] is not None:
                ax.scatter(self.params["ref_pnt_disp"][0], self.params["ref_pnt_disp"][1], c='r',s=70)
            #plt.show()
            plt.tight_layout()
            fig.savefig('nds_' + str(self.params["current_iteration_count"])
                        + '_' + str(0) + '.pdf')
            plt.clf()
            print('Plotted!')
        elif np.shape(obj_arch)[1] > 2:

            """
            fig_obj=plt_int.animate_parallel_coords_init_(data=obj_arch,
                                                  data_unc=unc_arch,
                                                  vectors=edge_adapted_vectors,
                                                  preference=self.params["ref_pnt_disp"],
                                                  filename= str(self.params["current_iteration_count"])+"_test_manyobj.html")
            
            fig_obj=animate_init_(data=obj_arch,
                                  filename=str(self.params["current_iteration_count"])+"_test_manyobj.html")
            """
            unc_avg_all = np.mean(unc_arch_all, axis=1)
            unc_avg_all_max = np.max(unc_avg_all)
            unc_avg_all_min = np.min(unc_avg_all)
            plt_int.plot_vals(objs=obj_arch_all,
                               unc=unc_arch_all,
                               preference=self.params["ref_pnt_disp"],
                               iteration=self.params["current_iteration_count"],
                               min=unc_avg_all_min,
                               max=unc_avg_all_max,
                               interaction_count=-1,
                               ideal=population.ideal_fitness,
                               nadir=population.nadir_fitness)

            plt_int.plot_vals(objs=obj_arch,
                               unc=unc_arch,
                               preference=self.params["ref_pnt_disp"],
                               iteration=self.params["current_iteration_count"],
                               interaction_count=0,
                               min=unc_avg_all_min,
                               max=unc_avg_all_max,
                               ideal=population.ideal_fitness,
                               nadir=population.nadir_fitness)

            plt_int.plot_vals(objs=obj_arch2,
                               unc=unc_arch2,
                               preference=self.params["ref_pnt_disp"],
                               iteration=self.params["current_iteration_count"],
                               interaction_count=-2,
                               min=unc_avg_all_min,
                               max=unc_avg_all_max,
                               ideal=population.ideal_fitness,
                               nadir=population.nadir_fitness)

        print("Total solutions after pre-filtering:")
        print(np.shape(obj_arch)[0])
        end_disp = 1.0
        end_disp = float(input("Do you want to SET thresholds : "))

        while end_disp > 0.0:
            no_solns = True
            while no_solns is True:
                for index in range(len(thresholds)):
                    while True:
                        print("Set the uncertainty threshold for objective (in %)", index + 1)
                        print("Minimum uncertainty value % = ", min_uncertainty[index])
                        print("Maximum uncertainty value % = ", max_uncertainty[index])
                        thresh_val = input("Please input the uncertainty threshold :  ")
                        print(thresh_val)
                        thresh_val = float(thresh_val)
                        print(thresh_val)
                        if thresh_val > min_uncertainty[index]:
                            thresholds[index] = thresh_val
                            break
                #loc = np.where(thresholds > unc_arch_percent)
                #loc = np.where((thresholds[0] > unc_arch_percent[:, 0]) & (thresholds[1] > unc_arch_percent[:, 1]))
                loc = np.where(np.all(np.tile(thresholds,(np.shape(unc_arch_percent)[0],1))> unc_arch_percent, axis=1))
                if np.size(loc)>0:
                    no_solns = False
                else:
                    print("No solutions! Please re-enter preferences.")
            count_interaction_thresh += 1

            np.savetxt('Obj_arch_cutoff' + '_' + str(self.params["current_iteration_count"])
                       + '_' + str(count_interaction_thresh) + '.csv', obj_arch[loc],
                       delimiter=",")
            np.savetxt('Unc_arch_cutoff' + '_' + str(self.params["current_iteration_count"])
                       + '_' + str(count_interaction_thresh) + '.csv', unc_arch[loc],
                       delimiter=",")
            np.savetxt('Indiv_arch_cutoff' + '_' + str(self.params["current_iteration_count"])
                       + '_' + str(count_interaction_thresh) + '.csv', indiv_arch[loc],
                       delimiter=",")

            if np.shape(obj_arch)[1] == 2:
                fig = plt.figure(1, figsize=(6, 6))
                ax = fig.add_subplot(111)
                ax.set_xlabel('$f_1$')
                ax.set_ylabel('$f_2$')
                plt.xlim(min_range_x, max_range_x)
                plt.ylim(min_range_y, max_range_y)
                #unc_avg = np.mean(unc_arch, axis=1)
                #unc_avg = unc_avg / unc_avg_all_max
                #ax.scatter(obj_arch_all[:, 0], obj_arch_all[:, 1], c=unc_avg_all, alpha=0.3, vmax=1,
                #           vmin=unc_avg_all_min)
                ax.scatter(obj_arch_all[:, 0], obj_arch_all[:, 1], c='gray')
                #ax.errorbar(obj_arch[loc, 0], obj_arch[loc, 1], xerr=1.96 * unc_arch[loc, 0],
                #             yerr=1.96 * unc_arch[loc, 1],
                #             fmt='o', ecolor='g')
                ax.scatter(obj_arch[loc, 0], obj_arch[loc, 1], c=np.reshape((unc_avg[loc]),(1,-1)), vmax=1, vmin=unc_avg_all_min)
                if self.params["ref_pnt_disp"] is not None:
                    ax.scatter(self.params["ref_pnt_disp"][0], self.params["ref_pnt_disp"][1], c='r',s=70)
                #plt.show()
                plt.tight_layout()
                fig.savefig('threshold_'+ str(self.params["current_iteration_count"])
                            + '_' + str(count_interaction_thresh) + '.pdf')
                print('Plotted!')
            elif np.shape(obj_arch)[1] > 2:
                plt_int.plot_vals(objs=obj_arch[loc],
                                   unc=unc_arch[loc],
                                   preference=self.params["ref_pnt_disp"],
                                   iteration=self.params["current_iteration_count"],
                                   interaction_count=count_interaction_thresh,
                                   min=unc_avg_all_min,
                                   max=unc_avg_all_max,
                                   ideal=population.ideal_fitness,
                                   nadir=population.nadir_fitness)
                """
                fig_obj = plt_int.animate_parallel_coords_next_(data=obj_arch[loc],
                                                      data_unc=unc_arch[loc],
                                                      vectors=edge_adapted_vectors,
                                                      preference=self.params["ref_pnt_disp"],
                                                      filename=str(self.params["current_iteration_count"])+"_test_manyobj.html",
                                      figure=fig_obj,
                                      generation=count_interaction_thresh)
                
                fig_obj = animate_next_(data=obj_arch[loc],
                                        figure=fig_obj,
                                        generation=count_interaction_thresh,
                                        filename=str(self.params["current_iteration_count"])+"_test_manyobj.html")
                """

            end_disp = float(input("Do you want to reset thresholds : "))
            print(end_disp)


    def _run_interruption(self, population: "Population"):
        """Run the interruption phase of RVEA.

        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors, conducts interaction with the user.

        Parameters
        ----------
        population : Population
        """
        if self.params["interact"] or (
            self.params["a_priori"] and self.params["current_iteration_count"] == 1
        ):

            self.plot_solutions2(population)
            # refpoint = np.mean(population.fitness, axis=0)
            ideal = population.ideal_fitness
            nadir = population.nadir_fitness
            refpoint = np.zeros_like(ideal)
            print("Ideal vector is ", ideal)
            print("Nadir vector is ", nadir)
            #for index in range(int(len(refpoint)/2)):
            for index in range(len(refpoint)):
                while True:
                    print("Preference for objective ", index + 1)
                    print("Ideal value = ", ideal[index])
                    print("Nadir value = ", nadir[index])
                    pref_val = float(
                        input("Please input a value between ideal and nadir: ")
                    )
                    if pref_val > ideal[index] and pref_val < nadir[index]:
                        refpoint[index] = pref_val
                        break
            self.params["ref_pnt_disp"] = refpoint
            refpoint = refpoint - ideal
            norm = np.sqrt(np.sum(np.square(refpoint)))
            refpoint = refpoint / norm
            print(refpoint)
            self.params["ref_pnt_normalized"] = refpoint
            self.params["reference_vectors"].iteractive_adapt_1(refpoint)
            #self.params["reference_vectors"].interactive_adapt_6(refpoint)
            self.params["reference_vectors"].add_edge_vectors()
            self.params["fmin_iteration"] = np.amin(population.objectives,axis=0)
        else:
            print("Adapting...")
            #non_dom_front = ndx(population.objectives_archive)
            #non_dom_fitness = population.objectives_archive[non_dom_front[0][0]]
            #population.objectives_archive = population.objectives_archive[non_dom_front[0][0]]
            #population.individuals_archive = population.individuals_archive[non_dom_front[0][0]]
            #self.params["reference_vectors"].adapt(non_dom_fitness)
            self.params["reference_vectors"].adapt(population.fitness)
        self.params["reference_vectors"].neighbouring_angles()

    def select(self, population: "Population"):
        """Describe a selection mechanism. Return indices of selected
        individuals.

        # APD Based selection. # This is different from the paper. #
        params.genetations != total number of generations. This is a compromise.
        Also this APD uses an archived ideal point, rather than current, potentially
        worse ideal point.

        Parameters
        ----------
        population : Population
            Population information

        Returns
        -------
        list
            list: Indices of selected individuals.
        """
        #term1 = (self.params["total_gen_count"] / (self.params["generations"] * self.params["iterations"]))
        term1 = population.func_evals/self.params["maximum_func_evals"]
        #if term1 > 1:
        #    term1 = 1
        penalty_factor = (term1
            ** self.params["Alpha"]
        ) * population.problem.num_of_objectives



        return APD_select(
            fitness=population.fitness,
            vectors=self.params["reference_vectors"],
            penalty_factor=penalty_factor,
            ideal=population.ideal_fitness,
            uncertainty=population.uncertainty,
            unc_rf=population.uncertainty_rf,
            unc_lpr=population.uncertainty_lpr
            #fmin_constant=self.params["fmin_iteration"]
        )

    def continue_evolution(self, population) -> bool:
        """Checks whether the current iteration should be continued or not."""
        return self.params["maximum_func_evals"] > population.func_evals

    def continue_iteration(self):
        """Checks whether the current iteration should be continued or not."""
        return self.params["current_iteration_gen_count"] <= self.params["generations"]

