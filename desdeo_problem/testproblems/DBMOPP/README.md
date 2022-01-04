# DBMOPP Generator

This DBMOPP generator is based on the original Distance-Based Multi-Objective Point Problem (DBMOPP) instance generator https://github.com/fieldsend/DBMOPP_generator.
Currently the usage and code is very similar to the original Matlab version.

## Example usage 

Example usage as notebook-tutorial at desdeo_problem/docs/notebooks/DBMOPP_tutorial.

## What works 
Combining DBMOPP with DESDEO's MOProblem instances for solving the generated
testproblems with DESDEO's methods is work in progress.

- Original DBMOPP generator's functionality is done.
- MOProblem can be called to evaluate objective function values.
- MOProblem can be called to evaluate constraints.
- Constraints include:
    - Hard / Soft vertex constraints
    - Hard / Soft center constraints
    - Hard / Soft moat constraints
    - Hard / Soft extended checker constraints

  - Can we approximation of the Pareto set either by:
    - calling the plot_pareto_set_members() which returns the found members.
    - calling get_Pareto_set_member() which returns one pareto set member uniformly from the pareto regions.

## What does not work yet

- Disconnected Pareto penalty regions are not yet plotted.
- get_Pareto_set_member currently only works for problems with 2 design variables.
- Some parameters and combination of them is not tested yet with MOProblem.
These are ndo, vary_sol_density, vary_objective_scales, prop_neutral and nm and
are suggested to be left on their default values.
