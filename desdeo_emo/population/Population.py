from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np

from desdeo_emo.population.CreateIndividuals import create_new_individuals
from desdeo_emo.recombination.BoundedPolynomialMutation import BP_mutation
from desdeo_emo.recombination.SimulatedBinaryCrossover import SBX_xover
from desdeo_problem.Problem import MOProblem


class BasePopulation(ABC):
    def __init__(self, problem: MOProblem, pop_size: int, pop_params: Dict = None):
        self.pop_size: int = pop_size
        self.problem = problem
        self.individuals: Union[List, np.ndarray] = None
        self.objectives: np.ndarray = None
        self.uncertainity: np.ndarray = None
        self.fitness: np.ndarray = None
        if not problem.n_of_constraints == 0:
            self.constraint = None
        self.ideal_objective_vector = problem.ideal
        self.nadir_objective_vector = problem.nadir
        self.ideal_fitness_val = None
        self.nadir_fitness_val = None
        if problem.ideal is not None:
            self.ideal_fitness_val = problem.ideal * problem._max_multiplier
            self.nadir_fitness_val = problem.nadir * problem._max_multiplier
        self.xover = None
        self.mutation = None
        self.recombination = None


    @abstractmethod
    def add(self, offsprings: Union[List, np.ndarray]) -> List:
        """Evaluate and add offspring to the population.

        Parameters
        ----------
        offsprings : Union[List, np.ndarray]
            List or array of individuals to be evaluated and added to the population.

        Returns
        -------
        List
            Indices of the evaluated individuals
        """
        pass

    @abstractmethod
    def keep(self, indices: List):
        """Save the population members given by the list of indices for the next
            generation. Delete the rest.

        Parameters
        ----------
        indices : List
            List of indices of the population members to be kept for the next
                generation.
        """
        pass

    @abstractmethod
    def delete(self, indices: List):
        """Delete the population members given by the list of indices for the next
            generation. Keep the rest.

        Parameters
        ----------
        indices : List
            List of indices of the population members to be deleted.
        """
        pass

    @abstractmethod
    def mate(
        self, mating_individuals: List = None, params: Dict = None
    ) -> Union[List, np.ndarray]:
        """Perform crossover and mutation over the population members.

        Parameters
        ----------
        mating_individuals : List, optional
            List of individuals taking part in recombination. By default None, which
                recombinated all individuals in random order.
        params : Dict, optional
            Parameters for the mutation or crossover operator, by default None.

        Returns
        -------
        Union[List, np.ndarray]
            The offspring population
        """
        pass


class Population(BasePopulation):
    def __init__(
        self,
        problem: MOProblem,
        pop_size: int,
        pop_params: Dict = None,
        use_surrogates: bool = False,
    ):
        super().__init__(problem, pop_size)
        self.lower_limits = self.problem.get_variable_lower_bounds()
        self.upper_limits = self.problem.get_variable_upper_bounds()
        self.individuals_archive = {}
        self.objectives_archive = {}
        self.uncertainty_archive = {}
        self.gen_count = 1
        self.individuals_archive[str(self.gen_count)] = self.individuals
        self.objectives_archive[str(self.gen_count)] = self.objectives
        self.uncertainty_archive[str(self.gen_count)] = self.uncertainity
        if pop_params is None:
            design = "LHSDesign"
        if pop_params is not None:
            if "design" in pop_params.keys():
                design = pop_params["design"]
            else:
                design = "LHSDesign"
        individuals = create_new_individuals(design, problem, pop_size)
        self.add(individuals, use_surrogates)
        self.xover = SBX_xover()
        self.mutation = BP_mutation(self.lower_limits, self.upper_limits)

    def add(
        self, offsprings: Union[List, np.ndarray], use_surrogates: bool = False
    ) -> List:
        """Evaluate and add offspring to the population.

        Parameters
        ----------
        offsprings : Union[List, np.ndarray]
            List or array of individuals to be evaluated and added to the population.
        
        use_surrogates: bool
            If true, use surrogate models rather than true function evaluations.

        Returns
        -------
        List
            Indices of the evaluated individuals
        """
        results = self.problem.evaluate(offsprings, use_surrogates)
        objectives = results.objectives
        fitness = results.fitness
        constraints = results.constraints
        uncertainity = results.uncertainity
        self.individuals_archive[str(self.gen_count)] = offsprings
        self.objectives_archive[str(self.gen_count)] = objectives
        self.uncertainty_archive[str(self.gen_count)] = uncertainity
        self.gen_count += 1
        if self.individuals is None:
            self.individuals = offsprings
            self.objectives = objectives
            self.fitness = fitness
            self.constraint = constraints
            self.uncertainity = uncertainity
            first_offspring_index = 0
        else:
            first_offspring_index = self.individuals.shape[0]
            if self.individuals.ndim - offsprings.ndim == 1:
                self.individuals = np.vstack((self.individuals, [offsprings]))
            elif self.individuals.ndim == offsprings.ndim:
                self.individuals = np.vstack((self.individuals, offsprings))
            else:
                pass  # TODO raise error
            self.objectives = np.vstack((self.objectives, objectives))
            self.fitness = np.vstack((self.fitness, fitness))
            if self.problem.n_of_constraints != 0:
                self.constraint = np.vstack(
                    (self.constraint, constraints)
                )
            if uncertainity is None:
                self.uncertainity = None
            else:    
                self.uncertainity = np.vstack((self.uncertainity, uncertainity))
        last_offspring_index = self.individuals.shape[0]
        self.update_ideal()
        return list(range(first_offspring_index, last_offspring_index))

    def keep(self, indices: List):
        """Save the population members given by the list of indices for the next
            generation. Delete the rest.

        Parameters
        ----------
        indices : List
            List of indices of the population members to be kept for the next
                generation.
        """
        #print("size=",self.individuals.shape[0])
        mask = np.zeros(self.individuals.shape[0], dtype=bool)
        mask[indices] = True
        self.individuals = self.individuals[mask]
        self.objectives = self.objectives[mask]
        self.fitness = self.fitness[mask]
        if self.uncertainity is not None:
            self.uncertainity = self.uncertainity[mask]
        if self.problem.n_of_constraints > 0:
            self.constraint = self.constraint[mask]
        

    def delete(self, indices: List):
        """Delete the population members given by the list of indices for the next
            generation. Keep the rest.

        Parameters
        ----------
        indices : List
            List of indices of the population members to be deleted.
        """
        mask = np.ones(self.individuals.shape[0], dtype=bool)
        mask[indices] = False
        self.individuals = self.individuals[mask]
        if len(self.individuals) == 0:
            self.individuals = None
        self.objectives = self.objectives[mask]
        self.fitness = self.fitness[mask]
        if self.uncertainity is not None:
            self.uncertainity = self.uncertainity[mask]
        if self.problem.n_of_constraints > 0:
            self.constraint = self.constraint[mask]

    def mate(self, mating_individuals: List = None) -> Union[List, np.ndarray]:
        """Perform crossover and mutation over the population members.

        Parameters
        ----------
        mating_individuals : List, optional
            List of individuals taking part in recombination. By default None, which
                recombinated all individuals in random order.
        params : Dict, optional
            Parameters for the mutation or crossover operator, by default None.

        Returns
        -------
        Union[List, np.ndarray]
            The offspring population
        """
        if self.recombination is not None:
            offspring = self.recombination.do(self.individuals, mating_individuals)
        else:
            offspring = self.xover.do(self.individuals, mating_individuals)
            offspring = self.mutation.do(offspring)
        return offspring

    def update_ideal(self):
        if self.ideal_fitness_val is None:
            self.ideal_fitness_val = np.amin(self.fitness, axis=0)
        else:
            self.ideal_fitness_val = np.amin(
                np.vstack((self.ideal_fitness_val, self.fitness)), axis=0
            )
        self.ideal_objective_vector = (
            self.ideal_fitness_val * self.problem._max_multiplier
        )
