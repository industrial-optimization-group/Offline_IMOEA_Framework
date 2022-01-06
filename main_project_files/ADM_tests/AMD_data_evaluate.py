import matlab.engine
import pickle
import numpy as np
from desdeo_problem.testproblems.TestProblems import test_problem_builder

eng = matlab.engine.start_matlab()
s = eng.genpath('./matlab_files')
eng.addpath(s, nargout=0)

def evaluate_population(init_folder,
                        path_to_file,
                        problem_testbench, 
                        problem_name, 
                        nobjs, 
                        nvars, 
                        sampling, 
                        nsamples, 
                        approaches,
                        run):


    s = eng.genpath(init_folder)
    eng.addpath(s, nargout=0)
    path_to_file = path_to_file + '/Run_' + str(run)
    data = pickle.load(open(path_to_file, "rb"))
    data_evaluted = {}
    for i in range(len(data)):
        data_evaluted[i] = {}
        for approach in approaches:
            pop_obj_temp = data[i][approach]
            population = pop_obj_temp.individuals
            size_pop = np.shape(population)[0]
            if problem_testbench == 'DTLZ':
                prob = test_problem_builder(
                            name=problem_name, n_of_objectives=nobjs, n_of_variables=nvars
                        )
                np_a = prob.evaluate(population)[0]
            elif problem_testbench == 'DDMOPP':
                population = matlab.double(population.tolist())
                objs_evaluated = eng.evaluate_python(population, init_folder, sampling, problem_name, nobjs, nvars, nsamples, 0, 0)
                #print(objs_evaluated)
                np_a = np.array(objs_evaluated._data.tolist())
                np_a = np_a.reshape((nobjs,size_pop))
                np_a = np.transpose(np_a)
            data_evaluted[i][approach] = np_a
    #print(data_evaluted)
    outfile = open(path_to_file + '_evaluated', 'wb')
    pickle.dump(data_evaluted, outfile)
    outfile.close()