nsamples = 109;
Mobj = [5];

num_dimensions=7
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

load Initial_Population_DDMOPP_LHS_AM_109.mat
for m = 1:length(Mobj)
    M=Mobj(m);
    num_objectives=M
    problem_parameters = distance_problem_generator(num_objectives,num_dimensions,...
    curvature, number_of_disconnected_set_regions,...
    number_of_local_fronts,number_of_dominance_resistance_regions, ...
    number_of_discontinuous_regions,...
    varying_density,non_identical_pareto_sets,varying_objective_ranges, ...
    fill_space,plot_wanted,random_seed)
    save(strcat('DDMOPP_Params_',problem,'_',num2str(M),'.mat'),'problem_parameters');
    obj_vals = zeros(nsamples,num_objectives);
    for Run = 1:35

        Population = Initial_Population_DDMOPP(Run).c;
        for samp = 1:nsamples
            obj_vals(samp,:) = distance_points_problem(Population(samp,:),problem_parameters);        
        end
        %obj_vals
        Obj_vals_DDMOPP(Run).c = obj_vals;

    end

    Obj_vals_DDMOPP
    save(strcat('Obj_vals_DDMOPP_LHS_AM_109_5vars_',problem,'_',num2str(M),'.mat'),'Obj_vals_DDMOPP');
    
end