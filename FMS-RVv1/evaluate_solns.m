clear all;
close all;


plot_init=0
is_plot=0
plot_progress = 1;
write_initsamples=0;
write_all_samples=1;
write_all_samples_data=0;
write_final_solns=0
%Problems = {'DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7'};
% Problems = {'DTLZ1','DTLZ2','DTLZ3','DTLZ4','DTLZ5','DTLZ6','DTLZ7'};
%Problems = {'DTLZ2','DTLZ4','DTLZ5','DTLZ6','DTLZ7'};
%Problems = {'WFG1','WFG2','WFG3','WFG5','WFG9' };
%Problems = {'WFG1','WFG2','WFG3','WFG4','WFG5','WFG6','WFG7','WFG8','WFG9' };
%Problems = {'DTLZ2'};
%Problems = {'DTLZ2','DTLZ5'};
Problems = {'P1','P2'};
%Problems = {'P2'};

Benchmark = 'DDMOPP'; %,'DTLZ'};
%Benchmark = 'DTLZ';
%Algorithms = {'NSGAIII'}; %'IBEA'
Algorithms = {'RVEA'}; %'IBEA'
%Algorithms = {'IBEA'}; %'IBEA'

Mobj=[2,3,4,5,6,8,10]; %,5];
%Mobj = [2]%,5];
%Mobj=[2,3,5,8];
num_vars = [10];
sample_size = 109;
%managements = {'1','9','7','8'}; %,'Offline_m5_ibea','Offline_ei2_ibea'}; %'Offline_m3','Offline_m5','Offline_m6','Offline_m3_ei','Offline_m3_ei2'}; %,'Offline_m3','Offline_m4'}; %'Offline_m2','Offline_m1','Offline_m3'}; %,
%%managements = {'1','7','8'}; 
%managements = {'0','9','1','7','8','12','72','82'};
%managements = {'1','7','8','12','72','82'};
%managements = {'12','72','82'};
%managements = {'1','7'}
managements = {'0'}

Strategies = {'LHS','MVNORM'};
%Strategies = {'LHS'};
%Strategies = {'MVNORM'}
main_folder='Tests_Probabilistic_Rev2';
%main_folder='Tests_soln_plot'
%main_folder='Tests_CSC_2'
init_folder='~/Work/Codes/Offline_IMOEA_Framework/AM_Samples_109_Final';
run_folder='~/Work/Codes/Tests_Probabilistic_Rev2';
%run_folder='../Tests_soln_plot'
%run_folder='../Tests_CSC_2'

RunNum = 1;
%med_index=[2,9,4,4];

D = 10;
%  Mat=[TX,TY];


%addpath(genpath('PLOY'));
%addpath(genpath('RBF'));
addpath(genpath('DBMOPP_generator'));

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
                        Initial_Population_DDMOPP=load(fullfile(init_folder,['Initial_Population_DDMOPP_' Strategy '_AM_' num2str(nvars) '_' num2str(sample_size) '.mat']));
                        load(fullfile(init_folder,['Obj_vals_DDMOPP_' Strategy '_AM_'  Problem '_' num2str(M) '_' num2str(nvars) '_' num2str(sample_size) '.mat']));
                    else
                        load(fullfile(init_folder,['Initial_Population_DTLZ_' Strategy '_AM_' num2str(nvars) '_' num2str(sample_size) '.mat']));
                    end
                        %figure;
                    for mgmt = 1:length(managements)
                        
                        management = managements{mgmt};
                        
                        
                        folder=fullfile(run_folder,['Offline_Mode_' management '_' algorithm],Strategy,[Problem '_' num2str(M) '_' num2str(nvars)])
                        
                        if ~exist(folder, 'dir')
                            mkdir(folder);
                        end
                        for Run = 1:RunNum
                            if RunNum==1
                                filename = fullfile(folder,['med_indices.csv']) 
                                med_index_new=csvread(filename)
                                %Run=med_index(mgmt)+1;
                                Run = med_index_new(1,1)+1;
                            end
                            Run
                            
                            
                            if strcmp(management,'9')==1
                                if strcmp(Benchmark,'DDMOPP')==1
                                    run_data=load(fullfile(folder,['Run_' num2str(Run-1) '.mat']));
                                    run_data=run_data.run_data;
                                    %obj_vals=run_data.RBFval;
                                    %X = run_data.TX;

                                    if write_all_samples == 1
                                        X = run_data.TX;
                                        obj_vals_surr = run_data.RBFval;
                                    else
                                        obj_vals_surr = run_data.RBFval(end-100:end,:);
                                        X = run_data.TX(end-100:end,:);
                                    end
                                    %non = P_sort(obj_vals_surr,'first')==1;
                                    %non_dom_pop = X(non,:);
                                    %PF=obj_vals_surr(non,:);
                                    %population = run_data.TX(end-100:end,:);
                                    %population = non_dom_pop;
                                    population = X;
                                    %objvals = Obj_vals_DDMOPP(Run).c;
                                    if plot_init == 1
                                       population = Initial_Population_DDMOPP.Initial_Population_DDMOPP(Run+1).c;
                                    end
                                   obj_vals = zeros(size(population,1),M);
                                   for samp = 1:size(population,1)
                                        obj_vals(samp,:) = distance_points_problem(population(samp,:),problem_parameters);        
                                        %obj_vals
                                   end
                                   %obj_vals
                                   if plot_init == 0
                                       X = population;                                     
                                   else
                                       X = Initial_Population_DDMOPP.Initial_Population_DDMOPP(Run+1).c;
                                   end

                                else
                                    run_data=load(fullfile(folder,['Run_' num2str(Run-1) '.mat']));
                                    %obj_vals=run_data.RBFval;
                                    %X = run_data.TX;
                                    if write_all_samples == 1
                                        X = run_data.run_data.TX;
                                        obj_vals_surr = run_data.run_data.RBFval;
                                    else
                                        obj_vals_surr = run_data.run_data.RBFval(end-100:end,:);
                                        X = run_data.run_data.TX(end-100:end,:);
                                    end
                                    population = X;
                                    obj_vals = P_objective_v0('value',Problem,M,population);
                                                            
                                end
                            elseif strcmp(management,'0')==1
                                population = Initial_Population_DDMOPP.Initial_Population_DDMOPP(Run-1).c;
                                X=population
                            else
                               filename_obj=strcat(folder,'/','Run_', num2str(Run-1),'_surrx_all');
                               filename_pop=strcat(folder,'/', 'Run_', num2str(Run-1),'_popx_all');
                               obj_vals_surr =  dlmread(filename_obj);
                               population =  dlmread(filename_pop);
                               X=population;
                               size(population,1)
                               M
                               if strcmp(Benchmark,'DDMOPP')==1
                                   obj_vals = zeros(size(population,1),M);
                                       for samp = 1:size(population,1)
                                            obj_vals(samp,:) = distance_points_problem(population(samp,:),problem_parameters);        
                                            %obj_vals
                                       end
                               else
                                   obj_vals = P_objective_v0('value',Problem,M,population);
                               end
                            end
                               if plot_progress == 1
                                   if strcmp(management,'0')==1 
                                        population = Initial_Population_DDMOPP.Initial_Population_DDMOPP(Run-1).c;
                                        axis([-1,1,-1,1]);
                                        box on;
                                        axis square;
                                        set(gca,'xtick',[],'ytick',[]);
                                        set(gcf, 'PaperSize', [5 5])
                                        saveas(gcf,strcat(run_folder,'/ProgProj_',algorithm,'_',Strategy,'_',Problem,'_',num2str(M),'_',num2str(nvars),'_',num2str(mgmt)),'pdf');               
                                   elseif strcmp(management,'9')==1 
                                       load(fullfile(folder,['Run_' num2str(Run-1) '.mat']));
                                       obj_vals_surr = run_data.RBFval(end-100:end,:);
                                       population = run_data.TX(end-100:end,:);                                       
                                   else
                                       filename_obj=strcat(folder,'/','Run_', num2str(Run-1),'_obj');
                                       filename_pop=strcat(folder,'/', 'Run_', num2str(Run-1),'_pop');
                                       objs =  dlmread(filename_obj);
                                       population =  dlmread(filename_pop);                                       
                                   end
                                   obj_vals = zeros(size(population,1),M);
                                   for samp = 1:size(population,1)
                                    obj_vals(samp,:) = distance_points_problem(population(samp,:),problem_parameters); 
                                   end
                                   %figure;
                                   X_final=population;
                                   %obj_vals
                                   
                                     
                                        if nvars > 2
                                            X_project = zeros(size(X,1),2);
                                            for i=1:size(X,1)
                                                X_project(i,:) = project_nD_point_to_2D(X(i,:),problem_parameters.projection_vectors(1,:),problem_parameters.projection_vectors(2,:));
                                            end
                                            X= X_project;
                                        end
                                        %subplot(1,5,mgmt)
                                    if mgmt == 1
                                        figure;
                                        plot_dbmopp_2D_regions(problem_parameters,0,M,0, ...
                                                                0); 
                                        box on;
                                        set(gca,'xTick',[]);
                                        set(gca,'yTick',[]);
                                        xlabel('');
                                        ylabel('');
                                        set(gcf, 'PaperSize', [5 5])
                                        saveas(gcf,strcat(run_folder,'/ProgProj_',algorithm,'_',Strategy,'_',Problem,'_',num2str(M),'_',num2str(nvars),'_',num2str(-1)),'pdf');
                                    end
                                    %hold on;
                                    sz = 25;
                                    c = linspace(1,10,length(X));
                                    length(X)
                                    figure;
                                    if strcmp(management,'0')==1 
                                        %%%%%%%%%%%%% change here to plot
                                        %%%%%%%%%%%%% the instance and init
                                        %%%%%%%%%%%%% together
                                        %plot_dbmopp_2D_regions(problem_parameters,0,M,0, ...
                                        %                        0);
                                                            
                                        
                                        box on;
                                        set(gca,'xTick',[]);
                                        set(gca,'yTick',[]);
                                        xlabel('');
                                        ylabel('');hold on;
                                        
                                        scatter(X(:,1),X(:,2),sz,'yellow','filled');
                                    else
                                        scatter(X(:,1),X(:,2),sz,c,'filled');
                                    end
                                    hold on;
                                    %scatter(
                                    if nvars > 2
                                        X_project = zeros(size(X_final,1),2);
                                        for i=1:size(X_final,1)
                                            X_project(i,:) = project_nD_point_to_2D(X_final(i,:),problem_parameters.projection_vectors(1,:),problem_parameters.projection_vectors(2,:));
                                        end
                                        X= X_project;
                                    end
                                    
                                    non = P_sort(obj_vals,'first')==1;
                                    non_dom_pop = X(non,:);
                                    PF=obj_vals(non,:);
                                    scatter(non_dom_pop(:,1), non_dom_pop(:,2),'+','red','linewidth',2)
                                    %hold off;
                                    axis([-1,1,-1,1]);
                                    box on;
                                    axis square;
                                    set(gca,'xtick',[],'ytick',[]);
                                    set(gcf, 'PaperSize', [5 5])
                                    saveas(gcf,strcat(run_folder,'/ProgProj_',algorithm,'_',Strategy,'_',Problem,'_',num2str(M),'_',num2str(nvars),'_',management),'pdf');
                               end
                               
                               if write_initsamples == 1
                                non = P_sort(obj_vals,'first')==1;
                                %non
                                non_dom_pop = X(non,:);
                                PF=obj_vals(non,:);                               
                               end
                               
                               if write_all_samples_data == 1
                                   filename_solns = strcat(folder,'/','Run_', num2str(Run),'_soln_all');
                                   dlmwrite(filename_solns,obj_vals);
                                   filename_solns = strcat(folder,'/','Run_', num2str(Run),'_surr_all');
                                   dlmwrite(filename_solns,obj_vals_surr);
                                   filename_solns = strcat(folder,'/','Run_', num2str(Run),'_pop_all');
                                   dlmwrite(filename_solns,population);
                               elseif write_final_solns == 1
                                   filename_solns = strcat(folder,'/','Run_', num2str(Run),'_soln');
                                   dlmwrite(filename_solns,obj_vals);
                                   filename_solns = strcat(folder,'/','Run_', num2str(Run),'_surr');
                                   dlmwrite(filename_solns,obj_vals_surr);
                                   filename_solns = strcat(folder,'/','Run_', num2str(Run),'_pop');
                                   dlmwrite(filename_solns,population);
                                  % population
                               end
                            
                        end

                           
                           
                        

                    end
                    %set(gcf, 'PaperPosition', [0 0 25 5]); %Position plot at left hand corner with width 5 and height 5.

                end
            end
            close all;
        end
    end
end
close all;