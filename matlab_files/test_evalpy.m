clear all;
close all;
is_plot = 0;
plot_init = 0;
write_initsamples=0;
%Problems = {'DTLZ4','DTLZ5','DTLZ6','DTLZ7'};
% Problems = {'DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7'};
%Problems = {'DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7'};
%Problems = {'WFG1','WFG2','WFG3','WFG5','WFG9' };

%Problems = {'WFG1','WFG2','WFG3','WFG4','WFG5','WFG6','WFG7','WFG8','WFG9' };
%Problems = {'DTLZ7' };

%Problems = {'DTLZ2','DTLZ5'};
%Problems = {'P1','P2'};
%Problems = {'P1','P2','P3','P4'};
Problems = {'P1'};
%
%Algorithms = {'NSGAIII'}; %'IBEA'
Algorithms = {'RVEA'}; %'IBEA'
%Algorithms = {'IBEA'}; %'IBEA'
%Mobj=[3,5,6,8,10]; %,5];
%Mobj = [3,5,7];
Mobj = [3];
num_vars = [2]; %,8,10];
sample_sizes = [2000];
%managements = {'0','1','2'}; %,'Offline_m5_ibea','Offline_ei2_ibea'}; %'Offline_m3','Offline_m5','Offline_m6','Offline_m3_ei','Offline_m3_ei2'}; %,'Offline_m3','Offline_m4'}; %'Offline_m2','Offline_m1','Offline_m3'}; %,
%managements = {'7'};
%managements = {'generic_fullgp','generic_sparsegp','strategy_2','strategy_3'};
%managements = {'htgp'};
%managements = {'generic_fullgp','generic_sparsegp','htgp'}
%managements = {'generic_sparsegp','htgp'}
managements = {'generic_sparsegp'}
%managements = {'generic_fullgp'}
%Strategies = {'LHS','MVNORM'};
Strategies = {'LHS'};
%Strategies = {'MVNORM'}
main_folder='Test_DR_CSC_Final_1';
init_folder='../data/initial_samples';
run_folder='../data/test_runs';
%main_folder = 'Tests_additional_obj1sts_Probabilistic_Finalx_new
%main_folder='Offline_Prob'
%labx={'Full GP','Sparse GP','Strategy 2'};
labx={'Sparse GP'};
RunNum = 1;

for ss = 1:length(sample_sizes)
    sample_size = sample_sizes(ss);
    for algo = 1:length(Algorithms)
        algorithm = Algorithms{algo};
        for m = 1:length(Mobj)
            M=Mobj(m);
            for nv = 1:length(num_vars)
                nvars = num_vars(nv);
                for Prob = 1:length(Problems)
                    Problem = Problems{Prob};
                    for strat = 1:length(Strategies)
                        Strategy=Strategies{strat};
                        for mgmt = 1:length(managements)
                            %figure;
                            management = managements{mgmt};
                            folder=fullfile(run_folder,main_folder,['Offline_Mode_' management '_' algorithm],Strategy,num2str(sample_size),'DDMOPP',[Problem '_' num2str(M) '_' num2str(nvars)])
                            for Run = 1:RunNum
                                Run=Run-1
                                load(fullfile(folder,['Run_' num2str(Run) '.mat']))
                                population = run_data.individuals_solutions;
                                format long
                                objs = run_data.obj_solutions;
                                objs_evaluated = evaluate_python(population, init_folder, Strategy, Problem, M, nvars, sample_size, is_plot, plot_init)
                                %print(objs_evaluated)
                            end
                        end
                    end
                end
            end
        end
    end
end

                           