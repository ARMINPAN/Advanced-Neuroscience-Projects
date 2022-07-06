function [p_before_startingPoint, p_after_startingPoint, choice] = ...
    simple_model2(bias, sigma, start_point, dt, time_interval)

    mu = bias*dt;
    sigmaa = sigma*sqrt(dt);
    p_before_startingPoint = normcdf(start_point,mu,sigmaa)
    p_after_startingPoint = 1 - p_before_startingPoint;
    
    choice = rand < p_before_startingPoint;
end