clear all;
close all;
is_plot = 1;
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
Problems = {'P2'}%,'P2','P3','P4'};

%Algorithms = {'NSGAIII'}; %'IBEA'
Algorithms = {'RVEA'}; %'IBEA'
%Algorithms = {'IBEA'}; %'IBEA'
%Mobj=[3,5,6,8,10]; %,5];
Mobj = [5]%,5,7];
num_vars = [10]; %,8,10];
sample_size = 50000;
%managements = {'0','1','2'}; %,'Offline_m5_ibea','Offline_ei2_ibea'}; %'Offline_m3','Offline_m5','Offline_m6','Offline_m3_ei','Offline_m3_ei2'}; %,'Offline_m3','Offline_m4'}; %'Offline_m2','Offline_m1','Offline_m3'}; %,
%managements = {'7'};
%managements = {'init_pop','generic_fullgp','generic_spasegp'};
%managements = {'init_pop','generic_fullgp','generic_sparsegp','htgp_mse'};
managements = {'init_pop','generic_sparsegp','htgp'}
Strategies = {'LHS'}%,'MVNORM'};
%Strategies = {'LHS'};
%Strategies = {'MVNORM'}
%main_folder='Test_DR_CSC_1';
main_folder='Test_DR_CSC_Final_1';
init_folder='../data/initial_samples';
run_folder='../data/test_runs';
%main_folder = 'Tests_additional_obj1sts_Probabilistic_Finalx_new
%main_folder='Offline_Prob'
%labx={'Full GP','Spasrse GP'};
labx = {'init_pop','generic_sparsegp','htgp'}
RunNum = 1;


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
                load(fullfile(init_folder,['DDMOPP_Params_' Strategy '_' Problem '_' num2str(M) '_' num2str(nvars) '_' num2str(sample_size) '.mat']))
                load(fullfile(init_folder,['Initial_Population_DDMOPP_' Strategy '_AM_' num2str(nvars) '_' num2str(sample_size) '.mat']));
                figure;
                title(strcat(algorithm,'_',Strategy,'_',Problem, '_', num2str(M), '_', num2str(nvars)));
                for mgmt = 1:length(managements)              
                    management = managements{mgmt};
                    folder=fullfile(run_folder,main_folder,['Offline_Mode_' management '_' algorithm],Strategy,num2str(sample_size),'DDMOPP',[Problem '_' num2str(M) '_' num2str(nvars)])
                    
                    for Run = 1:RunNum
                       Run=Run-1
                       if mgmt ~= 1
                           load(fullfile(folder,['Run_' num2str(Run) '.mat']))
                           population = run_data.individuals_solutions;
                           objs = run_data.obj_solutions;
                           %filename_obj=strcat(folder,'/','Run_', num2str(Run),'_obj');
                           %filename_pop=strcat(folder,'/', 'Run_', num2str(Run),'_pop');
                           %    objs =  dlmread(filename_obj);
                           %population =  dlmread(filename_pop);
                           obj_vals = zeros(size(population,1),M);
                           
                       else
                               population = Initial_Population_DDMOPP(Run+1).c;
                               obj_vals = zeros(size(population,1),M);
                       end

                           for samp = 1:size(population,1)
                                obj_vals(samp,:) = distance_points_problem(population(samp,:),problem_parameters);        
                                %obj_vals
                           end

                           if mgmt ~= 1
                               X = population;                                     
                           else
                               X = Initial_Population_DDMOPP(Run+1).c;
                           end

                               if is_plot == 1
                                    if nvars > 2
                                        X_project = zeros(size(X,1),2);
                                        for i=1:size(X,1)
                                            X_project(i,:) = project_nD_point_to_2D(X(i,:),problem_parameters.projection_vectors(1,:),problem_parameters.projection_vectors(2,:));
                                        end
                                        X= X_project;
                                    end
 
                                    subplot(1,4,mgmt)
                                    plot_dbmopp_2D_regions(problem_parameters,0,M,0, ...
                                                            0);

                                    hold on;
                                    scatter(X(:,1),X(:,2));
                                    %scatter(
                                    non = P_sort(obj_vals,'first')==1;
                                    non
                                    non_dom_pop = X(non,:);
                                    PF=obj_vals(non,:);
                                    scatter(non_dom_pop(:,1), non_dom_pop(:,2),'*')
                                    hold off;
                                    
                               end
                       if mgmt == 1  
                            non = P_sort(obj_vals,'first')==1;
                            non
                            non_dom_pop = X(non,:);
                            PF=obj_vals(non,:);                               
                       end
        
                    end
                    
                end
                title(strcat(Strategy,',',Problem, ',', num2str(M), ',', num2str(nvars)));
                
                
                %set(gcf, 'PaperPosition', [0 0 20 5]); %Position plot at left hand corner with width 5 and height 5.
                %set(gcf, 'PaperSize', [20 5])
                %saveas(gcf,'MyPDFFileName','pdf');
                x1 = linspace(-1,1);
                x2 = linspace(-1,1);
                [X1,X2] = meshgrid(x1,x2);
                population(1,:)
                size(X1)
                for i = 1:100
                    for j=1:100
                        Z = distance_points_problem([X1(i,j),X2(i,j)],problem_parameters);        
                        Z1(i,j) = Z(1);
                        Z2(i,j) = Z(2);
                        if M == 3
                            Z3(i,j) = Z(3);
                        end
                    end
                end
                figure;
                
                contour(X1,X2,Z1)
                hold on;
                contour(X1,X2,Z2)
                if M==3
                    contour(X1,X2,Z3)
                end
                hold off;
                title(strcat(Strategy,',',Problem, ',', num2str(M), ',', num2str(nvars)));
            end
        end
    end    
end
end
