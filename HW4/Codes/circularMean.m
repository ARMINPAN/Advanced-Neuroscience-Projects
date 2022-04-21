function meanTheta = circularMean(theta)
    theta = rad2deg(theta);
    theta = reshape(theta,[1 size(theta,1)*size(theta,2)]);
    meanY = 0;
    meanX = 0;
    
    for i = 1:length(theta)
        if (~isnan(theta(i)))
            meanY = meanY + sind(theta(i));
            meanX = meanX + cosd(theta(i));
        end
    end
    
    meanTheta = atan2(meanY,meanX);
end