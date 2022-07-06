function speed = calSpeed(py,px,diff)

    distanceElecs = 400*10^-6; % in m
    Fs = 200;
    
    % numerator of the speed
    speedNum = abs(mean(diff(~isnan(diff))*Fs));
 
    % denominator of the speed  
    grad2 = sqrt((px/distanceElecs).^2 + (py/distanceElecs).^2);
    speedDenom = mean((mean(grad2(~isnan(grad2)))));

    speed = speedNum/speedDenom*100; % cm/s
end