# from desdeo_emo.othertools.ReferenceVectors import ReferenceVectors
import numpy as np
import adm_emo.baseADM as baseADM


def generateRP4learning(base: baseADM):

    ideal_cf = base.ideal_point

    translated_cf = base.translated_front

    # Assigment of the solutions to the vectors
    assigned_vectors = base.assigned_vectors

    # Find the vector which has a minimum number of assigned solutions
    number_assigned = np.bincount(assigned_vectors)
    min_assigned_vector = np.atleast_1d(
        np.squeeze(
            np.where(
                number_assigned == np.min(number_assigned[np.nonzero(number_assigned)])
            )
        )
    )
    sub_population_index = np.atleast_1d(
        np.squeeze(np.where(assigned_vectors == min_assigned_vector[0]))
        # If there are multiple vectors which have the minimum number of solutions, first one's index is used
    )
    # Assigned solutions to the vector which has a minimum number of solutions
    sub_population_fitness = translated_cf[sub_population_index]
    # Distances of these solutions to the origin
    sub_pop_fitness_magnitude = np.sqrt(
        np.sum(np.power(sub_population_fitness, 2), axis=1)
    )
    # Index of the solution which has a minimum distance to the origin
    minidx = np.where(sub_pop_fitness_magnitude == np.nanmin(sub_pop_fitness_magnitude))
    distance_selected = sub_pop_fitness_magnitude[minidx]

    # Create the reference point
    reference_point = distance_selected[0] * base.vectors.values[min_assigned_vector[0]]
    #print("ref1:",reference_point)
    #print("ideal:",ideal_cf)
    reference_point = np.squeeze(reference_point + ideal_cf)
    # reference_point = reference_point + ideal_cf
    return reference_point


def get_max_assigned_vector(assigned_vectors):

    #print('Assigned vectors:',assigned_vectors)
    number_assigned = np.bincount(assigned_vectors)
    #print('num ass:',number_assigned)
    max_assigned_vector = np.atleast_1d(
        np.squeeze(
            np.where(
                number_assigned == np.max(number_assigned[np.nonzero(number_assigned)])
            )
        )
    )
    #print('max ass:',max_assigned_vector)
    return max_assigned_vector


def generateRP4decision(base: baseADM, max_assigned_vector):

    assigned_vectors = base.assigned_vectors
    #print('Assigned vectors D:',assigned_vectors)
    ideal_cf = base.ideal_point

    translated_cf = base.translated_front
    #print('Assigned vectors:', assigned_vectors)
    #print('Translated cf:', translated_cf)
    sub_population_index = np.atleast_1d(
        np.squeeze(np.where(assigned_vectors == max_assigned_vector))
    )
    #print("Sub pop index D size:", np.shape(sub_population_index)[0])
    sub_population_fitness = translated_cf[sub_population_index]
    # Distances of these solutions to the origin
    sub_pop_fitness_magnitude = np.sqrt(
        np.sum(np.power(sub_population_fitness, 2), axis=1)
    )
    # Index of the solution which has a minimum distance to the origin
    #print("sub pop fit size:",np.shape(sub_pop_fitness_magnitude)[0])
    try:
        minidx = np.where(sub_pop_fitness_magnitude == np.nanmin(sub_pop_fitness_magnitude))
    except Exception as e:
        print('Exception:', e)
        print("sub pop fit ma:",sub_pop_fitness_magnitude)
        print('Assigned vectors:', assigned_vectors)
        print('Translated cf:', translated_cf)
        print('max ass vect:', max_assigned_vector)
<<<<<<< HEAD
    
=======

>>>>>>> 5d77055adb5f5ce50cea938c21cd3d4ea97b1671
    distance_selected = sub_pop_fitness_magnitude[minidx]

    # Create the reference point
    reference_point = distance_selected[0] * base.vectors.values[max_assigned_vector]
    reference_point = np.squeeze(reference_point + ideal_cf)
    # reference_point = reference_point + ideal_cf
    return reference_point
