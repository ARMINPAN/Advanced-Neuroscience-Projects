function pgd = AlterPGD(realPhis,fittedPhis)
    
    % num
    num  = 0;
    for i = 1:length(realPhis)
        if(~isnan(realPhis(i)))
            num = num + sind(realPhis(i)-nanmean(nanmean(realPhis)))*...
                sind(fittedPhis(i)-nanmean(nanmean(fittedPhis)));
        end
    end
    
    % denom
    a1  = 0;
    a2 = 0;
    for i = 1:length(realPhis)
        if(~isnan(realPhis(i)))
            a1 = a1 + sind(realPhis(i)-nanmean(nanmean(realPhis))).^2;
            a2 = a2 + sind(fittedPhis(i)-nanmean(nanmean(fittedPhis))).^2;
        end
    end
    denom = sqrt(a1*a2);
    
    
    roCC = num/denom;
    
    %%% alternate PGD
    
    pgd = 1 - (1-roCC^2)*(5*10-2-1)/(5*10-2-3-1);

end