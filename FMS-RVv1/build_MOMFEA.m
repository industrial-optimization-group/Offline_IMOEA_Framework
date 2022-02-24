function [GModel,FModel]=build_MOMFEA(TX,TY,M)
for m2 = 1:M
    PRmodel=POLY(TX,TY(:,m2),'quad');
    [RBFmodel, time] = rbfbuild(TX,TY(:,m2), 'TPS');
    GModel{m2}   = PRmodel;
    FModel{m2}   = RBFmodel;
end
                   
end