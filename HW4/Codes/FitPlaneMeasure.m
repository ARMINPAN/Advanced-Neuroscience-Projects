function fittingMeasure = FitPlaneMeasure(realPhis,fittedPhis)
    
    % cos
    a = 0;
    % a
    for i = 1:5
        for j = 1:10
            if ~isnan(realPhis(i,j))
                a = a + (cosd(realPhis(i,j)-fittedPhis(i,j)));
            end
        end
    end
    
    a = (a/50).^2;
    
    % sin
    b = 0;
    for i = 1:5
        for j = 1:10
            if ~isnan(realPhis(i,j))
                b = b + (sind(realPhis(i,j)-fittedPhis(i,j)));
            end
        end
    end
    
    b = (b/50).^2;
        
    
    %%% fittingMeasure 
   
    fittingMeasure = sqrt(a+b);
end