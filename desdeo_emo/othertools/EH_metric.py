import numpy as np
from warnings import warn
from typing import List, Callable
from desdeo_emo.selection.SelectionBase import SelectionBase
from desdeo_emo.population.Population import Population
from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors



class Expanding_HC():
    """ This is the expanding hypercube sampling that is suited for evaluating 
    a set of solutions that are created based on a reference point"""

    def __init__(
        self
    ):
        pass

    def hypercube_size(self, ref, sets, num_of_objectives,a):
        """Calculates the hypercube size for each solution set."""
        asf = [0] * num_of_objectives
        set_sizes = []
        current_solutions = sets
        #a= np.max(current_solutions, axis=0)
        b = np.min(current_solutions, axis=0)
        w = a
        #all_solutions = current_archive
        #w = [1]* num_of_objectives#in case of desirability
        if ref is None:
            ref = [0] * num_of_objectives
        
        #vectorize version
        tmp = (current_solutions - np.transpose(ref))/w
        max_vector = np.max (tmp, axis =1 )
        sort_max_vector = np.sort(max_vector)
        #ascending_max_vector = sort_max_vector[::-1]

        set_sizes.append(sort_max_vector)

        #index = min_sum_asf_index
        return set_sizes

    def area(self, set_sizes):
        """ line 13 in the algorithm"""
        #n = n_show
        i = 0
        ps = 0
        a= 0
        setAreas = []
        setAreas_final = []
        setSizes = []
        for key in set_sizes:
            s_sizes = set_sizes[key][0]
            #print(s_sizes)
            n_s_size = np.shape(s_sizes)[0]
            
            #print(n_s_size)
            i = 0
            a = 0
            ps = 0
            for s in s_sizes:
                a += (i/n_s_size) * (np.abs(s - ps))
                #print(a)
                i += 1
                ps = s
            setAreas.append(a)
            setSizes.append(ps)
        #print(setSizes)
        # print(setAreas)
        maxSize = np.max(np.asarray(setSizes))
        for setArea, setSize in zip(setAreas, setSizes):
            delta_area = maxSize - setSize
            setAreas_final.append(setArea + delta_area)

        return setAreas_final

