%  Mat the offline dataset
% M the number of objectives
function MFEA_data = main(Mat,M)
%  Mat=[TX,TY];
NP=100; % Population size for task 1
rmp=0.7; % Random mating probability
gen = 40; % Maximum Number of generations
TL = 15;
        len = size(Mat,2);
            
            TX = Mat(:,1:len-M);
			TY = Mat(:,1+len-M:end);
            for m = 1:M
                PRmodel=POLY(TX,TY(:,m),'quad');
                [RBFmodel, time] = rbfbuild(TX,TY(:,m), 'TPS');
                GModel{m}   = PRmodel;
                FModel{m}   = RBFmodel;
            end
            
            K = 10;
            MFEA_data = MOMFEA_v01(NP,rmp,gen,GModel,FModel,lb,ub,D,Mat,problem,M,TX,TY,TL);

end