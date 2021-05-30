clear; clc; 

addpath(genpath('DBMOPP_generator'));

plot_dir = 'Plots_Atanu_Prob_Paper';
n_local = 0;
n_dom = 0;

for fe = 1:2
    if fe==1
        n_obj = 6;
    elseif fe==2
        n_obj = 10;
    end
%% read and plot the problem structure

%% plot the problem
plot_dbmopp_2D_regions(distance_problem_parameters,0,n_obj,n_local,n_dom);
box on;
set(gca,'xTick',[]);
set(gca,'yTick',[]);
xlabel('');
ylabel('');
saveas(gcf,[plot_dir '/Problem_feature_' num2str(fe)],'epsc');


%% load the data - Initial sampling

dec_var_initial = Initial_samples(:,1:end-n_obj);
dec_var_generic = Generic(:,1:end-n_obj);
dec_var_transfer = Transfer(:,1:end-n_obj);
dec_var_prob = Prob(:,1:end-n_obj);
dec_var_hybrid = Hybrid(:,1:end-n_obj);

%%
obj_initial = Initial_samples(:,end-n_obj+1:end);
obj_generic = Generic(:,end-n_obj+1:end);
obj_transfer = Transfer(:,end-n_obj+1:end);
obj_prob = Prob(:,end-n_obj+1:end);
obj_hybrid = Hybrid(:,end-n_obj+1:end);



plot_trace_generation(dec_var_initial,obj_initial,distance_problem_parameters);
saveas(gcf,[plot_dir '/Solutions_Initial_feature_' num2str(fe)],'epsc');
plot_trace_generation(dec_var_generic,obj_generic,distance_problem_parameters);
saveas(gcf,[plot_dir '/Solutions_Generic_feature_' num2str(fe)],'epsc');
plot_trace_generation(dec_var_transfer,obj_transfer,distance_problem_parameters);
saveas(gcf,[plot_dir '/Solutions_Transfer_feature_' num2str(fe)],'epsc');
plot_trace_generation(dec_var_prob,obj_prob,distance_problem_parameters);
saveas(gcf,[plot_dir '/Solutions_Prob_feature_' num2str(fe)],'epsc');
plot_trace_generation(dec_var_hybrid,obj_hybrid,distance_problem_parameters);
saveas(gcf,[plot_dir '/Solutions_Hybrid_feature_' num2str(fe)],'epsc');
end

function plot_trace_generation(dec_var,obj,distance_problem_parameters)

nvars = size(dec_var,2);
if nvars > 2
    n1 = size(dec_var,1); 
    dec_var_2d = zeros(n1,2); 
    for i=1:n1
        dec_var_2d(i,:) = project_nD_point_to_2D(dec_var(i,:),distance_problem_parameters.projection_vectors(1,:),distance_problem_parameters.projection_vectors(2,:));
    end
else
    dec_var_2d = dec_var;
end

%% enter the number of solutions in the last generation
non = P_sort(obj(:,end-100:end),'first')==1;
dec_var_2d_non = dec_var_2d(non,:);

n_fun = size(dec_var_2d,1);
c = parula(n_fun);
figure;
scatter(dec_var_2d(:,1),dec_var_2d(:,2),[],c,'filled');
hold on;
scatter(dec_var_2d_non(:,1),dec_var_2d_non(:,2),2,'+','black','LineWidth',12);
hold off;
axis([-1,1,-1,1]);
box on;
axis square;
set(gca,'xtick',[],'ytick',[]);

end