function [LIP_thresh_evidence, p_LIP, t, dt, LIP_events, MT_E_events, MT_I_events] ...
    = lip_activity(MT_p_values,LIP_weights,LIP_threshold)
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
    rate = 0.0;
    LIP_event_times = [];
    MT_E_events = [];
    MT_I_events = [];
    t = 0;
    i = 1;
    
    while rate < LIP_threshold  
        t = (i-1)*dt;
        dN = rand(1,2) < MT_p_values;
        N = N + dN;
        
        % MT neurons events
        if(dN(1) == 1)
            MT_E_events = [MT_E_events t];
        end
        if(dN(2) == 1)
            MT_I_events = [MT_I_events t];
        end
        
        p_LIP(i) = sum(N.*LIP_weights);
        LIP_thresh_evidence = 5;
        LIP_event = LIP_thresh_evidence < p_LIP(i);
        
        % LIP neuron activation
        if(LIP_event)
            LIP_event_times = [LIP_event_times t];
        end
        
        % check LIP mean rate for last M spikes
        if(length(LIP_event_times) > 100)
            M = 100;
            Denom = LIP_event_times(length(LIP_event_times)-M);
            rate = M/(t - Denom);
            [t rate]
        end
        i = i+1; 
    end
    LIP_events = LIP_event_times;
end