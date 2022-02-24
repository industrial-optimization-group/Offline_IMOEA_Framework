import Main_Execute_ADM as mexeADM
import pickle
import pickle_to_mat_converter as pickmat
#from AMD_data_evaluate import evaluate_population
import os
from joblib import Parallel, delayed
import datetime
import traceback
<<<<<<< HEAD:main_project_files/Start_All_ADM.py

=======
>>>>>>> 5d77055adb5f5ce50cea938c21cd3d4ea97b1671:main_project_files/ADM_tests/Start_All_ADM.py

data_folder = '/scratch/project_2003769/Codes/data'
#main_directory = 'Offline_Prob_DDMOPP3'
#main_directory = 'Tests_Probabilistic_Finalx_new'
#main_directory = 'Tests_additional_obj1'
#main_directory = 'Tests_Gpy_1'
#main_directory = 'Tests_new_adapt'
#main_directory = 'Tests_toys'
#main_directory = 'Test_Gpy3'
#main_directory = 'Test_DR_4'  #DR = Datatset Reduction
<<<<<<< HEAD:main_project_files/Start_All_ADM.py
main_directory = 'Test_Interactive_new1'
#main_directory = 'Test_DR_CSC_1'
=======
#main_directory = 'Test_DR_Scratch'
#main_directory = 'Test_Interactive_2'
main_directory = 'Test_Interactive_new1'
>>>>>>> 5d77055adb5f5ce50cea938c21cd3d4ea97b1671:main_project_files/ADM_tests/Start_All_ADM.py
#main_directory = 'Test_RF'
#main_directory = 'Test_DR_CSC_Final_1'
init_folder = data_folder + '/initial_samples_109'


<<<<<<< HEAD:main_project_files/Start_All_ADM.py
file_exists_check = True
=======
#file_exists_check = True
file_exists_check = False
>>>>>>> 5d77055adb5f5ce50cea938c21cd3d4ea97b1671:main_project_files/ADM_tests/Start_All_ADM.py
#convert_to_mat = True
evaluate_data = True
#evaluate_data = False
#import Telegram_bot.telegram_bot_messenger as tgm
#dims = [5,8,10] #,8]
#dims = [2, 5, 7, 10]
dims = [10]
#dims = [27]

sample_sizes = [109]
#sample_sizes = [2000, 10000, 50000]
#dims = 4
############################################
#folder_data = 'AM_Samples_109_Final'
#folder_data = 'AM_Samples_1000'


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



<<<<<<< HEAD:main_project_files/Start_All_ADM.py
#objectives = [3, 5, 7, 9]
#objectives = [3,5,7]
objectives = [5,9]
#objectives = [5, 7, 9]
=======
objectives = [5,9]
#objectives = [5,7,9]
#objectives = [3, 5, 7]
>>>>>>> 5d77055adb5f5ce50cea938c21cd3d4ea97b1671:main_project_files/ADM_tests/Start_All_ADM.py
#objectives = [3,5,7]
#objectives = [2,3,5]
#objectives = [2,3,4,5,6,8,10]
#objectives = [3,5,6,8,10]
#objectives = [3,5,6,8,10]

<<<<<<< HEAD:main_project_files/Start_All_ADM.py
problems = ['DTLZ4','DTLZ7']
=======
#problem_testbench = 'DTLZ'
problem_testbench = 'DDMOPP'
#problem_testbench = 'GAA'

#problems = ['DTLZ4']
>>>>>>> 5d77055adb5f5ce50cea938c21cd3d4ea97b1671:main_project_files/ADM_tests/Start_All_ADM.py
#problems = ['DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7']

problems = ['P1']
#problems = ['P1','P5']
#problems = ['P1','P2','P3','P4','P5']
#problems = ['P2','P3','P4','P5']
#problems = ['P1','P3','P4']


#problems = ['WELDED_BEAM'] #dims=4
#problems = ['TRUSS2D'] #dims=3
#problems = ['GAA']

#modes = [1, 2, 3]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2, 7 = Approach_Prob
#modes = [7]  # 1 = Generic, 2 = Approach 1 , 3 = Approach 2
#modes = [1, 7, 8]
#approaches = ["generic_fullgp","generic_sparsegp"]
#approaches = ["generic_fullgp","generic_sparsegp","strategy_1"]
#approaches = ["generic_fullgp"]
#approaches = ["generic_sparsegp"]
#approaches = ["strategy_1"]
#approaches = ["strategy_2"]
#approaches = ["strategy_3"]
#approaches = ["rf"]
#approaches = ["htgp"]
#approaches = ["generic_sparsegp"]
#approaches = ["generic_fullgp","htgp"]
#approaches = ["generic_fullgp","generic_sparsegp","htgp"]
#approaches = ["generic_sparsegp","htgp"]
#approaches = ["genericRVEA","probRVEA"]
approaches = ["genericRVEA_0","probRVEA_0","genericRVEA_1","probRVEA_1"]
#approaches = ["probRVEA"]
approaches_string = '_'.join(approaches)

#sampling = ['BETA', 'MVNORM']
#sampling = ['LHS']
#sampling = ['BETA','OPTRAND','MVNORM']
#sampling = ['OPTRAND']
<<<<<<< HEAD:main_project_files/Start_All_ADM.py
#sampling = ['MVNORM']
sampling = ['LHS', 'MVNORM']
=======
sampling = ['MVNORM']
#sampling = ['LHS', 'MVNORM']
>>>>>>> 5d77055adb5f5ce50cea938c21cd3d4ea97b1671:main_project_files/ADM_tests/Start_All_ADM.py

#emo_algorithm = ['RVEA','IBEA']
emo_algorithm = ['RVEA']
#emo_algorithm = ['IBEA']
#emo_algorithm = ['NSGAIII']
#emo_algorithm = ['MODEL_CV']

interactive = True

#############################################


<<<<<<< HEAD:main_project_files/Start_All_ADM.py
nruns = 5
n_plus=12
parallel_jobs = 64
=======
nruns = 11
parallel_jobs = 2
>>>>>>> 5d77055adb5f5ce50cea938c21cd3d4ea97b1671:main_project_files/ADM_tests/Start_All_ADM.py
log_time = str(datetime.datetime.now())


def parallel_execute(run, algo, prob, n_vars, obj, samp, sample_size):
    run=run+n_plus
    path_to_file = data_folder + '/test_runs/'+  main_directory \
                + '/Offline_Mode_' + approaches_string + '_' + algo + \
                '/' + samp + '/' + str(sample_size) + '/' + problem_testbench  + '/' + prob + '_' + str(obj) + '_' + str(n_vars)
    print(path_to_file)
    with open(data_folder + '/test_runs/'+ main_directory+"/log_"+log_time+".txt", "a") as text_file:
        text_file.write("\n"+path_to_file+"___"+str(run)+"___Started___"+str(datetime.datetime.now()))
    if not os.path.exists(path_to_file):
        os.makedirs(path_to_file)
        print("Creating Directory...")

    if evaluate_data is False:
        if (problem_testbench == 'DTLZ' and obj < n_vars) or problem_testbench == 'DDMOPP':
            path_to_file = path_to_file + '/Run_' + str(run)
            if os.path.exists(path_to_file) is False or file_exists_check is False:
<<<<<<< HEAD:main_project_files/Start_All_ADM.py
=======
                print('Starting Run!')
>>>>>>> 5d77055adb5f5ce50cea938c21cd3d4ea97b1671:main_project_files/ADM_tests/Start_All_ADM.py
                try:
                    results_dict = mexeADM.run_adm(problem_testbench=problem_testbench, 
                                                        problem_name=prob, 
                                                        nobjs=obj, 
                                                        nvars=n_vars, 
                                                        sampling=samp, 
                                                        nsamples=sample_size, 
                                                        is_data=True, 
                                                        approaches=approaches,
                                                        run=run)
                    outfile = open(path_to_file, 'wb')
                    pickle.dump(results_dict, outfile)
                    outfile.close()
                    with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
                            text_file.write("\n"+path_to_file+"___"+str(run)+"___Ended___"+str(datetime.datetime.now()))
                except Exception as e:
                    print(e)
                    with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
<<<<<<< HEAD:main_project_files/Start_All_ADM.py
                        text_file.write("\n"+ str(e) + "______" + traceback.format_exc()+ "________" + str(datetime.datetime.now()))       
            else:
                with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
                    text_file.write("\n"+path_to_file+"___"+str(run)+"___File already exists!___"+str(datetime.datetime.now()))
                    #print('File already exists!')
    else:
        print("None!")
        """
        evaluate_population(init_folder,
                        path_to_file,
                        problem_testbench, 
                        prob, 
                        obj, 
                        n_vars, 
                        samp, 
                        sample_size, 
                        approaches,
                        run)
        """
=======
                        text_file.write("\n"+path_to_file+"___"+str(run)+ "-Error-"+str(e) + "______" + traceback.format_exc()+ "________" + str(datetime.datetime.now()))   
                                
            else:
                with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
                    text_file.write("\n"+path_to_file+"___"+str(run)+"___File already exists!___"+str(datetime.datetime.now()))
                    print('File already exists!')
    else:
        try:
            evaluate_population(init_folder,
                            path_to_file,
                            problem_testbench, 
                            prob, 
                            obj, 
                            n_vars, 
                            samp, 
                            sample_size, 
                            approaches,
                            run)
        except Exception as e:
            print(e)
>>>>>>> 5d77055adb5f5ce50cea938c21cd3d4ea97b1671:main_project_files/ADM_tests/Start_All_ADM.py

    with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
        text_file.write("\n"+path_to_file+"___"+str(run)+"___Ended___"+str(datetime.datetime.now()))


try:
    temp = Parallel(n_jobs=parallel_jobs)(
        delayed(parallel_execute)(run, algo, prob, n_vars, obj, samp, sample_size)        
        for run in range(nruns)
        for algo in emo_algorithm
        for prob in problems
        for n_vars in dims
        for obj in objectives
        for samp in sampling
        for sample_size in sample_sizes)
<<<<<<< HEAD:main_project_files/Start_All_ADM.py
except Exception as e:
    print(e)
    with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
        text_file.write("\n"+ str(e) + "______" + str(datetime.datetime.now()))     
=======
#    for run in range(nruns):
#        parallel_execute(run, path_to_file)
except Exception as e:
    print(e)
    with open(data_folder + '/test_runs/'+main_directory+"/log_"+log_time+".txt", "a") as text_file:
        text_file.write("\n"+ str(e) + "______" + str(datetime.datetime.now()))   
>>>>>>> 5d77055adb5f5ce50cea938c21cd3d4ea97b1671:main_project_files/ADM_tests/Start_All_ADM.py


