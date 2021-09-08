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

Problems = {'P1'}

%Algorithms = {'NSGAIII'}; %'IBEA'
Algorithms = {'RVEA'}; %'IBEA'
%Algorithms = {'IBEA'}; %'IBEA'
%Mobj=[2,3,4,5,6,8,10]; %,5];
Mobj = [3,5,10];
num_vars = [10]; %,8,10];
%managements = {'100','700','800'}; %,'Offline_m5_ibea','Offline_ei2_ibea'}; %'Offline_m3','Offline_m5','Offline_m6','Offline_m3_ei','Offline_m3_ei2'}; %,'Offline_m3','Offline_m4'}; %'Offline_m2','Offline_m1','Offline_m3'}; %,
%managements = {'7000','8000'};
%managements = {'1','7','8','12','72'};
%managements = {'1','7','8'}
managements = {'7205'}
%managements = {'84','74'};
%managements = {'1','7','8','12','72','82'}
%managements = {'1','7','8','12','72'}%,'9'}
%managements = {'800'}
%Strategies = {'LHS','MVNORM'};
Strategies = {'LHS'};
%Strategies = {'MVNORM'}
init_folder='~/Work/Codes/Offline_IMOEA_Framework/AM_Samples_109_Final/';
%main_folder='~/Work/Codes/Tests_CSC_R2_Finalx';
main_folder='~/Work/Codes/data/test_runs/Tests_R3_Monte_Final';
%main_folder='Tests_CSC_4'
%main_folder = 'Tests_additional_obj1'
%main_folder='Offline_Prob'
labx={'Generic Approach','Approach 1'};
RunNum = 31;


for algo = 1:length(Algorithms)
    algorithm = Algorithms{algo};
for m = 1:length(Mobj)
    M=Mobj(m);
    for nv = 1:length(num_vars)
    nvars = num_vars(nv)
        for Prob = 1:length(Problems)
            Problem = Problems{Prob};
            for strat = 1:length(Strategies)
                Strategy=Strategies{strat};
                load(strcat(init_folder,'DDMOPP_Params_',Strategy,'_',Problem,'_',num2str(M),'_',num2str(nvars),'_109.mat'))
                load([init_folder 'Initial_Population_DDMOPP_' Strategy '_AM_' num2str(nvars) '_109.mat']);
                for mgmt = 1:length(managements)              
                    management = managements{mgmt};
                    folder=fullfile(main_folder,['Offline_Mode_' management '_' algorithm],Strategy,[Problem '_' num2str(M) '_' num2str(nvars)])
                    for Runx = 1:RunNum
                       if RunNum==1
                                filename = fullfile(folder,['med_indices.csv']) 
                                med_index_new=csvread(filename)
                                %Run=med_index(mgmt)+1;
                                Run = med_index_new(1,1)+1;
                       else
                           Run=Runx-1
                       
                       end
                            %Run=Run-1
                            Run
                       %Run=Runx-1
                       
                       filename_obj=strcat(folder,'/','Run_', num2str(Run),'_obj');
                       filename_pop=strcat(folder,'/', 'Run_', num2str(Run),'_pop');

                           objs =  dlmread(filename_obj);
                       population =  dlmread(filename_pop);
                                              size(objs)
                       size(population)
                       obj_vals = zeros(size(population,1),M);

                       if plot_init == 1
                           population = Initial_Population_DDMOPP(Run+1).c;
                           obj_vals = zeros(size(population,1),M);
                       end


                       for samp = 1:size(population,1)
                            obj_vals(samp,:) = distance_points_problem(population(samp,:),problem_parameters);        
                            %obj_vals
                       end

                       if plot_init == 0
                           X = population;                                     
                       else
                           X = Initial_Population_DDMOPP(Run+1).c;
                       end

                           if is_plot == 1
                               figure;
                                if nvars > 2
                                    X_project = zeros(size(X,1),2);
                                    for i=1:size(X,1)
                                        X_project(i,:) = project_nD_point_to_2D(X(i,:),problem_parameters.projection_vectors(1,:),problem_parameters.projection_vectors(2,:));
                                    end
                                    X= X_project;
                                end
                                %figure;
                                plot_dbmopp_2D_regions(problem_parameters,0,M,0, ...
                                                        0);

                                hold on;
                                scatter(X(:,1),X(:,2));
                                %scatter(
                                non = P_sort(obj_vals,'first')==1;
                                %non
                                non_dom_pop = X(non,:);
                                PF=obj_vals(non,:);
                                scatter(non_dom_pop(:,1), non_dom_pop(:,2),'*')
                                hold off;
                           end
                           if write_initsamples == 1
                            non = P_sort(obj_vals,'first')==1;
                            %non
                            non_dom_pop = X(non,:);
                            PF=obj_vals(non,:);                               
                           end
                       filename_solns = strcat(folder,'/','Run_', num2str(Run),'_soln');
                       dlmwrite(filename_solns,obj_vals);
                    end
                end
            end
        end
    end    
end
end
