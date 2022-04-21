function PGD = calPGD(py,px)
    
    % remove nans
    px(isnan(py)) = nan;
    py(isnan(px)) = nan;

    % numerator of the PGD
    sumx = nanmean(nanmean(px));
    sumy = nanmean(nanmean(py));
    thetaNum = sqrt(sumx^2 + sumy^2);
    
    % denomarator of the PGD
    grad2 = sqrt(px.^2 + py.^2);
    thetaDenom = nanmean(nanmean(grad2));
    
    % output pgd
    PGD = thetaNum/thetaDenom;
end