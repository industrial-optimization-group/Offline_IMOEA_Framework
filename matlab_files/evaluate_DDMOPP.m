function [y] = evaluate_DDMOPP(x)

num_objectives=5
num_dimensions=10
curvature=false
number_of_disconnected_set_regions=0
number_of_local_fronts=0
number_of_dominance_resistance_regions=0
number_of_discontinuous_regions=0
varying_density=false
non_identical_pareto_sets=false
varying_objective_ranges=false
fill_space=false
plot_wanted=false
random_seed=1

problem_parameters = distance_problem_generator(num_objectives,num_dimensions,...
    curvature, number_of_disconnected_set_regions,...
    number_of_local_fronts,number_of_dominance_resistance_regions, ...
    number_of_discontinuous_regions,...
    varying_density,non_identical_pareto_sets,varying_objective_ranges, ...
    fill_space,plot_wanted,random_seed);

y = distance_points_problem(x,problem_parameters);