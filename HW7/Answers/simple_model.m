function [evidence_vals, choice] = simple_model(bias, sigma, dt, time_interval)

    X = zeros(size(time_interval));
    meanOfW = 0;
    stdOfW = sqrt(dt);
    
    % numerical implementation of the equation
    for i = 1:length(time_interval)-1
        dW = normrnd(meanOfW,stdOfW);
        X(i+1) = X(i) + bias*dt + sigma*dW;
    end
    evidence_vals = X;
    choice = X(end) > 0;
end