from typing import Dict, Union

from desdeo_emo.EAs.BaseEA import BaseDecompositionEA, eaError
from desdeo_emo.population.Population import Population
from desdeo_emo.selection.APD_Select_constraints import APD_Select
from desdeo_emo.selection.Prob_APD_Select_v3_design import Prob_APD_select_v3_design  # superfast y considering mean APD

from desdeo_problem.Problem import MOProblem
import numpy as np
from desdeo_tools.interaction import (
    SimplePlotRequest,
    ReferencePointPreference,
    validate_ref_point_data_type,
    validate_ref_point_dimensions,
    validate_ref_point_with_ideal,
)

class RVEA(BaseDecompositionEA):

    def __init__(
        self,
        design_model,
        problem: MOProblem,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        alpha: float = 2,
        lattice_resolution: int = None,
        selection_type: str = None,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        time_penalty_component: Union[str, float] = None
        
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            a_priori=a_priori,
            interact=interact,
            use_surrogates=use_surrogates,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
        )
        self.time_penalty_component = time_penalty_component
        #self.thresh_size = thresh_size
        self.objs_interation_end = None
        self.unc_interaction_end = None
        self.iteration_archive_individuals = None
        self.iteration_archive_objectives = None
        self.iteration_archive_fitness = None
        self.iteration_archive_uncertainty = None

        self.design_samples = None
        self.design_samples_archive = None
        self.design_model = design_model
        self.thresh_size = 5 * problem.n_of_objectives

        time_penalty_component_options = ["original", "function_count", "interactive"]
        if time_penalty_component is None:
            if interact is True:
                time_penalty_component = "interactive"
            elif total_function_evaluations > 0:
                time_penalty_component = "function_count"
            else:
                time_penalty_component = "original"
        if not (type(time_penalty_component) is float or str):
            msg = (
                f"type(time_penalty_component) should be float or str"
                f"Provided type: {type(time_penalty_component)}"
            )
            eaError(msg)
        if type(time_penalty_component) is float:
            if (time_penalty_component <= 0) or (time_penalty_component >= 1):
                msg = (
                    f"time_penalty_component should either be a float in the range"
                    f"[0, 1], or one of {time_penalty_component_options}.\n"
                    f"Provided value = {time_penalty_component}"
                )
                eaError(msg)
            time_penalty_function = self._time_penalty_constant
        if type(time_penalty_component) is str:
            if time_penalty_component == "original":
                time_penalty_function = self._time_penalty_original
            elif time_penalty_component == "function_count":
                time_penalty_function = self._time_penalty_function_count
            elif time_penalty_component == "interactive":
                time_penalty_function = self._time_penalty_interactive
            else:
                msg = (
                    f"time_penalty_component should either be a float in the range"
                    f"[0, 1], or one of {time_penalty_component_options}.\n"
                    f"Provided value = {time_penalty_component}"
                )
                eaError(msg)
        self.time_penalty_function = time_penalty_function
        self.alpha = alpha
        self.selection_type = selection_type
        if self.design_model is not None:
            self.design_samples = np.asarray(self.design_model(self.population.individuals))
            self.design_samples = np.transpose(self.design_samples,(0,2,1))
        selection_operator = APD_Select(
            pop=self.population,
            time_penalty_function=self.time_penalty_function,
            alpha=alpha,
            selection_type=selection_type
        )
        self.selection_operator = selection_operator


    def _time_penalty_constant(self):
        """Returns the constant time penalty value.
        """
        return self.time_penalty_component

    def _time_penalty_original(self):
        """Calculates the appropriate time penalty value, by the original formula.
        """
        return self._current_gen_count / self.total_gen_count

    def _time_penalty_interactive(self):
        """Calculates the appropriate time penalty value.
        """
        return self._gen_count_in_curr_iteration / self.n_gen_per_iter

    def _time_penalty_function_count(self):
        """Calculates the appropriate time penalty value.
        """
        return self._function_evaluation_count / self.total_function_evaluations
    
    def manage_preferences(self, preference=None):
        """Run the interruption phase of EA.

        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors (adaptation), conducts interaction with the user.

        """
        if not isinstance(preference, (ReferencePointPreference, type(None))):
            msg = (
                f"Wrong object sent as preference. Expected type = "
                f"{type(ReferencePointPreference)} or None\n"
                f"Recieved type = {type(preference)}"
            )
            raise eaError(msg)
        if preference is not None:
            if preference.request_id != self._interaction_request_id:
                msg = (
                    f"Wrong preference object sent. Expected id = "
                    f"{self._interaction_request_id}.\n"
                    f"Recieved id = {preference.request_id}"
                )
                raise eaError(msg)
        if preference is None and not self._ref_vectors_are_focused:
            self.reference_vectors.adapt(self.population.fitness)
        if preference is not None:
            ideal = self.population.ideal_fitness_val
            #fitness_vals = self.population.ob
            refpoint_actual = (
                preference.response.values * self.population.problem._max_multiplier
            )
            refpoint = refpoint_actual - ideal
            norm = np.sqrt(np.sum(np.square(refpoint)))
            refpoint = refpoint / norm
            
            # evaluate alpha_k
            #cos_theta_f_k = self.reference_vectors.find_cos_theta_f_k(refpoint_actual, self.population, self.objs_interation_end, self.unc_interaction_end)
            # adapt reference vectors
            #self.reference_vectors.interactive_adapt_offline_adaptive(refpoint, cos_theta_f_k)
            
            self.reference_vectors.iteractive_adapt_1(refpoint)
            
            self.reference_vectors.add_edge_vectors()   
            
        self.reference_vectors.neighbouring_angles()
        if self.iteration_archive_objectives is None: 
            self.iteration_archive_individuals = self.population.individuals
            self.iteration_archive_objectives = self.population.objectives
            self.iteration_archive_fitness = self.population.fitness
            self.iteration_archive_uncertainty = self.population.uncertainity
            if self.design_model is not None:
                self.design_samples_archive = self.design_samples
        else:
            self.iteration_archive_individuals = np.vstack((self.iteration_archive_individuals, self.population.individuals))
            self.iteration_archive_objectives = np.vstack((self.iteration_archive_objectives, self.population.objectives))
            self.iteration_archive_fitness = np.vstack((self.iteration_archive_fitness, self.population.fitness))
            self.iteration_archive_uncertainty = np.vstack((self.iteration_archive_uncertainty, self.population.uncertainity))
            if self.design_model is not None:
                self.design_samples_archive = np.vstack((self.design_samples_archive, 
                                                                    self.design_samples))
    
    def _next_gen(self):
        size_pop = np.shape(self.population.objectives)[0]
        #print("Pop size before recombination: ",size_pop)
        if self.interact is True and size_pop <= self.thresh_size and self._gen_count_in_curr_iteration == 1:
            self.population.individuals = self.iteration_archive_individuals
            self.population.objectives = self.iteration_archive_objectives
            self.population.fitness = self.iteration_archive_fitness
            self.population.uncertainity = self.iteration_archive_uncertainty
            self.design_samples = self.design_samples_archive
            print("population injected in genration :",self._gen_count_in_curr_iteration)
        else:
            offspring = self.population.mate()  # (params=self.params)
            self.population.add(offspring, self.use_surrogates)
            if self.design_model is not None:
                self.design_samples = np.asarray(self.design_model(self.population.individuals))
                self.design_samples = np.transpose(self.design_samples,(0,2,1))
            self._function_evaluation_count += offspring.shape[0]
        #print("Pop size after recombination: ",np.shape(self.population.objectives)[0])
        selected = self._select(self.design_samples)
        #print("selected size:",np.shape(selected)[0])
        self.population.keep(selected)
        self._current_gen_count += 1
        self._gen_count_in_curr_iteration += 1
        print("FE=",self._function_evaluation_count)
        print("Size=",self.design_samples.shape)
    
    def _select(self, design_samples) -> list:
        return self.selection_operator.do(self.population,self.reference_vectors,design_samples)




class ProbRVEAv3(RVEA):
    def __init__(
        self,
        design_model,
        problem: MOProblem,
        population_size: int = None,
        population_params: Dict = None,
        initial_population: Population = None,
        alpha: float = 2,
        lattice_resolution: int = None,
        a_priori: bool = False,
        interact: bool = False,
        use_surrogates: bool = False,
        n_iterations: int = 10,
        n_gen_per_iter: int = 100,
        total_function_evaluations: int = 0,
        time_penalty_component: Union[str, float] = None,
    ):
        super().__init__(
            problem=problem,
            population_size=population_size,
            population_params=population_params,
            initial_population=initial_population,
            lattice_resolution=lattice_resolution,
            a_priori=a_priori,
            interact=interact,
            use_surrogates=use_surrogates,
            n_iterations=n_iterations,
            n_gen_per_iter=n_gen_per_iter,
            total_function_evaluations=total_function_evaluations,
            design_model = design_model
        )
        selection_operator = Prob_APD_select_v3_design(
            self.design_samples, self.population, self.time_penalty_function, alpha
        )
        self.selection_operator = selection_operator
        design_model = design_model

    def manage_preferences(self, preference=None):
        """Run the interruption phase of EA.

        Use this phase to make changes to RVEA.params or other objects.
        Updates Reference Vectors (adaptation), conducts interaction with the user.

        """
        if not isinstance(preference, (ReferencePointPreference, type(None))):
            msg = (
                f"Wrong object sent as preference. Expected type = "
                f"{type(ReferencePointPreference)} or None\n"
                f"Recieved type = {type(preference)}"
            )
            raise eaError(msg)
        if preference is not None:
            if preference.request_id != self._interaction_request_id:
                msg = (
                    f"Wrong preference object sent. Expected id = "
                    f"{self._interaction_request_id}.\n"
                    f"Recieved id = {preference.request_id}"
                )
                raise eaError(msg)
        if preference is None and not self._ref_vectors_are_focused:
            self.reference_vectors.adapt(self.population.fitness)
        if preference is not None:
            ideal = self.population.ideal_fitness_val
            #fitness_vals = self.population.ob
            refpoint_actual = (
                preference.response.values * self.population.problem._max_multiplier
            )
            refpoint = refpoint_actual - ideal
            norm = np.sqrt(np.sum(np.square(refpoint)))
            refpoint = refpoint / norm
            
            # evaluate alpha_k
            cos_theta_f_k = self.reference_vectors.find_cos_theta_f_k(refpoint_actual, self.population, self.objs_interation_end, self.unc_interaction_end)
            # adapt reference vectors
            self.reference_vectors.interactive_adapt_offline_adaptive(refpoint, cos_theta_f_k)
            
            #self.reference_vectors.iteractive_adapt_1(refpoint)
            
            self.reference_vectors.add_edge_vectors()
            #print("Shape RV:", np.shape(self.reference_vectors.values))
        
        self.reference_vectors.neighbouring_angles()
        if self.iteration_archive_objectives is None: 
            self.iteration_archive_individuals = self.population.individuals
            self.iteration_archive_objectives = self.population.objectives
            self.iteration_archive_fitness = self.population.fitness
            self.iteration_archive_uncertainty = self.population.uncertainity
        else:
            self.iteration_archive_individuals = np.vstack((self.iteration_archive_individuals, self.population.individuals))
            self.iteration_archive_objectives = np.vstack((self.iteration_archive_objectives, self.population.objectives))
            self.iteration_archive_fitness = np.vstack((self.iteration_archive_fitness, self.population.fitness))
            self.iteration_archive_uncertainty = np.vstack((self.iteration_archive_uncertainty, self.population.uncertainity))

