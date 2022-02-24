clear all;
close all;

%Problems = {'DTLZ4','DTLZ5','DTLZ6','DTLZ7'};
% Problems = {'DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7'};
%Problems = {'DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7'};
%Problems = {'WFG1','WFG2','WFG3','WFG5','WFG9' };

%Problems = {'WFG1','WFG2','WFG3','WFG4','WFG5','WFG6','WFG7','WFG8','WFG9' };
%Problems = {'DTLZ2' };

%Problems = {'DTLZ2','DTLZ5'};
Problems = {'P2'};

Benchmark = 'DDMOPP'; %,'DTLZ'};
%Benchmark = 'DTLZ';

%Algorithms = {'NSGAIII'}; %'IBEA'
Algorithms = {'RVEA'}; %'IBEA'
%Algorithms = {'IBEA'}; %'IBEA'
Mobj=[2,3,5,6,10];
%Mobj = [4,8];
num_vars = [10];
sample_size = 109;
%managements = {'0','1','2'}; %,'Offline_m5_ibea','Offline_ei2_ibea'}; %'Offline_m3','Offline_m5','Offline_m6','Offline_m3_ei','Offline_m3_ei2'}; %,'Offline_m3','Offline_m4'}; %'Offline_m2','Offline_m1','Offline_m3'}; %,
managements = {'9'};

Strategies = {'LHS','MVNORM'};
%Strategies = {'LHS'};
%Strategies = {'MVNORM'};
main_folder='Tests_Probabilistic_Finalx_new';
init_folder='../AM_Samples_109_Final';
run_folder='../Tests_Probabilistic_Finalx_new';

RunNum = 11;

%lb = ones(1,10)*-1;
lb = ones(1,10)*-1;

ub = ones(1,10);

D = 10;
%  Mat=[TX,TY];
NP=100*2; % Population size for task 1
maxFE = 40000;

rmp=0.7; % Random mating probability
gen = round(maxFE/(NP/2)); % Maximum Number of generations
TL = 15;

addpath(genpath('PLOY'));
addpath(genpath('RBF'));

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
                    if strcmp(Benchmark,'DDMOPP')==1
                        load(fullfile(init_folder,['DDMOPP_Params_' Strategy '_' Problem '_' num2str(M) '_' num2str(nvars) '_' num2str(sample_size) '.mat']))
                        load(fullfile(init_folder,['Initial_Population_DDMOPP_' Strategy '_AM_' num2str(nvars) '_' num2str(sample_size) '.mat']));
                        load(fullfile(init_folder,['Obj_vals_DDMOPP_' Strategy '_AM_'  Problem '_' num2str(M) '_' num2str(nvars) '_' num2str(sample_size) '.mat']));
                    else
                        load(fullfile(init_folder,['Initial_Population_DTLZ_' Strategy '_AM_' num2str(nvars) '_' num2str(sample_size) '.mat']));
                    end
                        %figure;
                    for mgmt = 1:length(managements)
                        %figure;
                        management = managements{mgmt};
                        folder=fullfile(run_folder,['Offline_Mode_' management '_' algorithm],Strategy,[Problem '_' num2str(M) '_' num2str(nvars)])
                        if ~exist(folder, 'dir')
                            mkdir(folder);
                        end
                        parfor Run = 1:RunNum
                            
                            %if strcmp(Benchmark,'DDMOPP')==1
                                population = Initial_Population_DDMOPP(Run).c;
                                objvals = Obj_vals_DDMOPP(Run).c;
                            %else
                            %    population = Initial_Population_DTLZ(Run).c;
                            %    objvals = P_objective_v0('value',Problem,M,population);
                            %end
                            TX = population;
                            TY = objvals;
                            [GModel,FModel]=build_MOMFEA(TX,TY,M);
                            %for m2 = 1:M
                            %    PRmodel=POLY(TX,TY(:,m2),'quad');
                            %    [RBFmodel, time] = rbfbuild(TX,TY(:,m2), 'TPS');
                            %    GModel{m2}   = PRmodel;
                            %    FModel{m2}   = RBFmodel;
                            %end
                           %K = 10;
                           Mat=[population,objvals]; 
                           runx_data(Run).c = MOMFEA_v01_V2(NP,rmp,gen,GModel,FModel,lb,ub,D,Mat,M,TX,TY,TL);

                           
                        end
                        for Run = 1:RunNum
                            run_data = runx_data(Run).c;
                            %obj_vals = run_data.RBFval(end-100:end,:);
                            %scatter(obj_vals(:,1),obj_vals(:,2));
                            
                            save(fullfile(folder,['Run_' num2str(Run-1) '.mat']),'run_data');
                            %parallelcoords(run_data.TX(1:100,:))
                            %figure
                            %parallelcoords()
                        end
                    end
                end
            end
        end
    end
end
