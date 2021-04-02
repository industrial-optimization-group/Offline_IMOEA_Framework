function hv = HVPI(FunctionValue,ref_point)
    N = size(FunctionValue,1);
    FunctionValue_temp = FunctionValue;
    Index_temp = [1:N]';
    hv = zeros(N,1);
    j=0;
    i = 1;
    Shell.c = [];
     while j < N
         Non = P_sort(FunctionValue_temp,'first')==1;
         FunctionValue = FunctionValue_temp(Non,:);
         Index = Index_temp(Non,:);
         Shell(i).c = [FunctionValue,Index];
         r = find(Non==1);
         FunctionValue_temp(r,:) = [];
         Index_temp(r,:) = [];
         j = j + size(FunctionValue,1);
         i = i+1;
     end
     
     for s = 1:size(Shell,2)
         
         if s == size(Shell,2) || size(Shell,2)==1
                ff1 = Shell(s).c(:,1:end-1);
                ind = Shell(s).c(:,end);
            for tt = 1:size(ff1,1)
                pf = ff1(tt,:);
                hv(ind(tt),:) = P_evaluate_hv('HV',pf ,ref_point);
            end
         
         else
         
            ff1 = Shell(s).c(:,1:end-1);
            ind = Shell(s).c(:,end);
            ff2 = Shell(s+1).c(:,1:end-1);
            for tt = 1:size(ff1,1)
                pf = [ff1(tt,:);ff2];
                hv(ind(tt),:) = P_evaluate_hv('HV',pf ,ref_point);
            end
         end
     end
end