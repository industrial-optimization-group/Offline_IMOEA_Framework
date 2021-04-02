clear all;


Mobj=[2,3]%,4,5,6,8,10]; %,5];
num_vars = [2] %,8,10];
%managements = {'1','7','8'}; %,'Offline_m5_ibea','Offline_ei2_ibea'}; %'Offline_m3','Offline_m5','Offline_m6','Offline_m3_ei','Offline_m3_ei2'}; %,'Offline_m3','Offline_m4'}; %'Offline_m2','Offline_m1','Offline_m3'}; %,
Strategies = {'MVNORM', 'LHS'};
Problems = {'P2'};
Runs=35;
%Design='LHS';
%Design='Random';
sample_size = 2000;

%load Initial_Population_DDMOPP_LHS_AM_109.mat
nsamples = sample_size;



for m = 1:length(Mobj)
    M=Mobj(m);
    for nv = 1:length(num_vars)
    nvars = num_vars(nv);
        for Prob = 1:length(Problems)
            Problem = Problems{Prob};

              for strat = 1:length(Strategies)
                Strategy=Strategies{strat};
                load(['../data/initial_samples/Initial_Population_DDMOPP_' Strategy '_AM_' num2str(nvars) '_' num2str(sample_size) '.mat'])
                load(strcat('../data/initial_samples/Obj_vals_DDMOPP_',Strategy,'_AM_',Problem,'_', num2str(M), '_', num2str(nvars),'_',num2str(sample_size),'.mat'));

                    parfor Run = 1:35
                        Population = Initial_Population_DDMOPP(Run).c;
                        Obj_vals = Obj_vals_DDMOPP(Run).c;
                        max_dist = pdist2(ones(1,M),-1*ones(1,M));
                        ref_point = max_dist*ones(1,M)
                        hvpi(Run).c = HVPI(Obj_vals,ref_point)
                        
                    end
                    save(strcat('../data/initial_samples/DDMOPP_HVPI_',Strategy,'_',Problem,'_',num2str(M),'_',num2str(nvars),'_',num2str(sample_size),'.mat'), 'hvpi')
                    
              end
        end
    end
end