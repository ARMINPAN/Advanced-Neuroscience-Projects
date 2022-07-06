function [X, reaction_time, choice] = ...
    two_choice_trial(pos_thresh, neg_thresh, dt, sigma, start_point, bias)

    meanOfW = 0;
    stdOfW = sqrt(dt);
    X = start_point;
    i = 1;
    
    % drift diff
    while (X(i) < pos_thresh && X(i) > neg_thresh)
        dW = normrnd(meanOfW,stdOfW);
        dx = bias*dt + sigma*dW;
        X(i+1) = X(i) + dx;
        i = i+1;
    end
    
    if(X(i) > pos_thresh)
        X(end) = pos_thresh;
    elseif(X(i) < neg_thresh)
        X(end) = neg_thresh;
    end
    
    % choice
    if(X(end) > 0)
        choice = 1;
    else
        choice = -1;
    end

    reaction_time = i*dt;
end