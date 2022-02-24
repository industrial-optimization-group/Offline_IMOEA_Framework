%% script to plot the solutions of visualizable test problems

addpath(genpath('DBMOPP_generator'));
close all;

%% provide the number of local fronts and dominance resistance regions
number_of_local_fronts = 0;
number_of_dominance_resistance_regions = 0;
%% load the problem file here
% examples - one with 2 variables another with 5 variables
load 'example_problem_var_2.mat';
% load 'example_problem_var_5.mat';

%% read the decision variables values to be plotted
nvars = 5; 
% nvars = 5;
X = rand(10,nvars); 

%%

if nvars > 2
    X_project = zeros(size(X,1),2);
    for i=1:size(X,1)
        X_project(i,:) = project_nD_point_to_2D(X(i,:),distance_problem_parameters.projection_vectors(1,:),distance_problem_parameters.projection_vectors(2,:));
    end
    X= X_project;
end
n_obj= distance_problem_parameters.num_objectives;
plot_dbmopp_2D_regions(distance_problem_parameters,0,num_objectives,number_of_local_fronts, ...
number_of_dominance_resistance_regions);
hold on;
scatter(X(:,1),X(:,2));
hold off;














