function [dt, LIP_thresh_evidence1, LIP_thresh_evidence2,...
    p_LIP1, p_LIP2, LIP_event_times1, LIP_event_times2, MT_1_events, MT_2_events] =...
    lip_activity_2(MT_p_values,LIP_weights)
    

    % Parameters:
    % MT_p_values - a vector with 2 elements, firing probabilities for the
    % excitatory and inhibitory neurons, resp.
    % LIP_weights - a length 2 vector of weighting factors for the evidence
    % from the excitatory (positive) and
    % inhibitory (negative) neurons
    % LIP_threshold - the LIP firing rate that represents the choice threshold criterion
    % use fixed time scale of 1 ms
    
    dt = 0.001;
    N = [0 0]; % plus is first, minus is second
    rate1 = 0.0;
    rate2 = 0.0;
    LIP_event_times1 = [];
    LIP_event_times2 = [];
    MT_1_events = [];
    MT_2_events = [];
    t = 0;
    
    for i = 1:length(MT_p_values)
        t = (i-1)*dt;
        
        % spike or not? - mt neurons
        dN = rand(1,2) < MT_p_values(:,i)';
        N = N + dN;
        
        % MT neurons events
        if(dN(1) == 1)
            MT_1_events = [MT_1_events t];
        end
        if(dN(2) == 1)
            MT_2_events = [MT_2_events t];
        end
        
        % evidence accumulated
        p_LIP1(i) = sum(N.*LIP_weights(1,:));
        p_LIP2(i) = sum(N.*LIP_weights(2,:));
        
        % evidence thresholds
        LIP_thresh_evidence1 = 5;
        LIP_thresh_evidence2 = 5;
        
        % spike or not? lip neurons
        LIP_event1 = LIP_thresh_evidence1 < p_LIP1(i);
        LIP_event2 = LIP_thresh_evidence2 < p_LIP2(i);
        
        % LIP neuron 1 activation
        if(LIP_event1)
            LIP_event_times1 = [LIP_event_times1 t];
        end
        
        % LIP neuron 2 activation
        if(LIP_event2)
            LIP_event_times2 = [LIP_event_times2 t];
        end
        
    end 

end