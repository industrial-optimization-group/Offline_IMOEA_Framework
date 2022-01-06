#import Main_Execute_Prob as mexeprob
#import Main_Execute_SS as mexe

import main_execute_interactive as mexe
import pickle
import other_tools.pickle_to_mat_converter as pickmat
import os
from joblib import Parallel, delayed
import datetime
import traceback

convert_to_mat = False
#convert_to_mat = False
#import Telegram_bot.telegram_bot_messenger as tgm
#dims = [5,8,10] #,8]
#dims = [10]
dims = [27]

sample_size = 1000
#sample_size = 109
#dims = 4
############################################
#folder_data = 'AM_Samples_109_Final'
folder_data = 'AM_Samples_1000'
data_folder = '/home/amrzr/Work/Codes/data'
#init_folder = data_folder + '/initial_samples_old'
init_folder = data_folder + '/AM_Samples_1000'

#problem_testbench = 'DTLZ'
#problem_testbench = 'DDMOPP'
problem_testbench = 'GAA'
"""
objs(1) = max_NOISE;
objs(2) = max_WEMP;
objs(3) = max_DOC;
objs(4) = max_ROUGH;
objs(5) = max_WFUEL;
objs(6) = max_PURCH;
objs(7) = -min_RANGE;
objs(8) = -min_LDMAX;
objs(9) = -min_VCMAX;
objs(10) = PFPF;
"""

main_directory = 'Test_interactive_GAA'

objectives = [11]
#objectives = [2,3]
#objectives = [2,3,5]
#objectives = [2,3,4,5,6,8,10]
#objectives = [3,5,6,8,10]
#objectives = [3,5,6,8,10]

#problems = ['DTLZ5']
problems = ['GAA']
#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']
#problems = ['P1','P2']
#problems = ['P1']
#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3

#approaches = ["interactive_uncertainty"]
approaches = ["interactive_uncertainty"]

sampling = ['LHS']
#sampling = ['MVNORM']
#sampling = ['LHS', 'MVNORM']

#emo_algorithm = ['RVEA','IBEA']
#emo_algorithm = ['RVEA']
emo_algorithm = ['ProbRVEA_1']
#emo_algorithm = ['IBEA']
#emo_algorithm = ['NSGAIII']
#emo_algorithm = ['MODEL_CV']

interactive = True

#############################################

nruns = 1
log_time = str(datetime.datetime.now())
#tgm.send(msg='Started testing @'+str(log_time))

for samp in sampling:
    for obj in objectives:
        for n_vars in dims:
            for prob in problems:
                for algo in emo_algorithm:
                    for approach in approaches:
                        path_to_file = data_folder + '/test_runs/' + main_directory \
                                       + '/Offline_Mode_' + approach + '_' + algo + \
                                       '/' + samp + '/' + str(sample_size) + '/' + problem_testbench  + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
                        print(path_to_file)
                        with open(data_folder+'/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
                            text_file.write("\n"+path_to_file+"_____"+str(datetime.datetime.now()))
                        if not os.path.exists(path_to_file):
                            os.makedirs(path_to_file)
                            print("Creating Directory...")

                        def parallel_execute(run, path_to_file):
                            if convert_to_mat is False:
                                results_dict = mexe.run_optimizer(problem_testbench=problem_testbench, 
                                                                    problem_name=prob, 
                                                                    nobjs=obj, 
                                                                    nvars=n_vars, 
                                                                    sampling=samp, 
                                                                    nsamples=sample_size, 
                                                                    is_data=True, 
                                                                    approach=approach,
                                                                    run=run,
                                                                    path=path_to_file)
                                path_to_file = path_to_file + '/Run_' + str(run)
                                outfile = open(path_to_file, 'wb')
                                pickle.dump(results_dict, outfile)
                                outfile.close()
                            else:
                                path_to_file = path_to_file + '/Run_' + str(run)
                                pickmat.convert(path_to_file, path_to_file+'.mat')



                        try:
                        #    temp = Parallel(n_jobs=nruns)(
                        #        delayed(parallel_execute)(run, path_to_file) for run in range(nruns))
                            for run in range(nruns):
                                parallel_execute(run, path_to_file)
                        #   tgm.send(msg='Finished Testing: \n' + path_to_file)
                        except Exception as e:
                        #    tgm.send(msg='Error occurred : \n' + path_to_file + '\n' + str(e))
                            print(str(e) + "______" + traceback.format_exc())        
                        #for run in range(nruns):
                        #    parallel_execute(run, path_to_file)
#tgm.send(msg='All tests completed successfully')

