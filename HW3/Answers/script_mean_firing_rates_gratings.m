%  This script contains three parts:
%   1. Convert spike times to 1ms bins.
%   2. Remove bad/very low firing units.
%   3. Compute the trial-averaged population activity (PSTHs).
%
%  S struct is used to store the data:
%    S(igrat).trial(itrial).spikes  (num_units x num_1ms_timebins)
%    S(igrat).trial(itrial).counts  (num_units x num_20ms_timebins)
%    S(igrat).mean_FRs  (num_units x num_20ms_timebins)
%
%  Author: Ben Cowley, bcowley@cs.cmu.edu, Oct. 2016
%
%  Notes:
%    - automatically saves 'S' in ./


%% parameters

    SNR_threshold = 1.5;
    firing_rate_threshold = 1.0;  % 1.0 spikes/sec
    binWidth = 20;  % 20 ms bin width

    
%% parameters relevant to experiment

    length_of_gratings = 1;  % each gratings was shown for 1.28s, take the last 1s
    
    filenames{1} = './data_monkey1_gratings.mat';
    filenames{2} = './data_monkey2_gratings.mat';
    filenames{3} = './data_monkey3_gratings.mat';


    monkeys = {'monkey1', 'monkey2', 'monkey3'};
    

%%  spike times --> 1ms bins

    for imonkey = 1:length(monkeys)
        S = [];

        fprintf('binning spikes for %s\n', monkeys{imonkey});

        load(filenames{imonkey});
            % returns data.EVENTS

        num_neurons = size(data.EVENTS,1);
        num_gratings = size(data.EVENTS,2);
        num_trials = size(data.EVENTS,3);

        edges = 0.28:0.001:1.28;  % take 1ms bins from 0.28s to 1.28s

        for igrat = 1:num_gratings
            for itrial = 1:num_trials
                for ineuron = 1:num_neurons
                    S(igrat).trial(itrial).spikes(ineuron,:) = histc(data.EVENTS{ineuron, igrat, itrial}, edges);
                end
                S(igrat).trial(itrial).spikes = S(igrat).trial(itrial).spikes(:,1:end-1);  % remove extraneous bin at the end
            end
        end

        save(sprintf('./S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end
    




%%  Pre-processing:  Remove bad/very low firing units

    % remove units based on low SNR
    removedNeurons{1} = [];
    removedNeurons{2} = [];
    removedNeurons{3} = [];

   

    for imonkey = 1:length(monkeys)
        load(filenames{imonkey});
            % returns data.SNR
        removedNeurons{imonkey} = [removedNeurons{imonkey}; find((data.SNR >= SNR_threshold) == 0)];
        %%%%%%%%%%%%%
        load(sprintf('./S_%s.mat', monkeys{imonkey}));
        S2 = S;
        num_grats = length(S2);
        num_trials = length(S2(1).trial);
        
        mean_FRs = [];   
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                mean_FRs = [mean_FRs sum(S2(igrat).trial(itrial).spikes,2)/1.0];
            end
        end
        mean_FRs_gratings = mean(mean_FRs,2);
        removedNeurons{imonkey} = [removedNeurons{imonkey}; find((mean_FRs_gratings >= firing_rate_threshold) == 0)];
       
        %%%%%%%%%%%%%
        keepNeurons = data.SNR >= SNR_threshold;
        clear data;
        
        fprintf('keeping units with SNRs >= %f for %s\n', SNR_threshold, monkeys{imonkey});
        
        load(sprintf('./S_%s.mat', monkeys{imonkey}));
        S2 = S;
        
            % returns S(igrat).trial(itrial).spikes
        
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                S(igrat).trial(itrial).spikes = S(igrat).trial(itrial).spikes(keepNeurons,:);
            end
        end
        
        save(sprintf('./S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
        removedNeurons{imonkey} = unique(removedNeurons{imonkey});
    end
    
    % remove units with mean firing rates < 1.0 spikes/sec
    
    for imonkey = 1:length(monkeys)
        load(sprintf('./S_%s.mat', monkeys{imonkey}));
            % returns S(igrat).trial(itrial).spikes
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        mean_FRs = [];   
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                mean_FRs = [mean_FRs sum(S(igrat).trial(itrial).spikes,2)/1.0];
            end
        end
        mean_FRs_gratings = mean(mean_FRs,2);
       % removedNeurons{imonkey} = [removedNeurons{imonkey}; find((mean_FRs_gratings >= firing_rate_threshold) == 0)];
        keepNeurons = mean_FRs_gratings >= firing_rate_threshold;
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                S(igrat).trial(itrial).spikes = S(igrat).trial(itrial).spikes(keepNeurons,:);
            end
        end
          
        save(sprintf('./S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end

    save('removedNeurons.mat', 'removedNeurons', '-v7.3');

%%  Take spike counts in bins
    for imonkey = 1:length(monkeys)
        
        fprintf('spike counts in %dms bins for %s\n', binWidth, monkeys{imonkey});
        
        load(sprintf('./S_%s.mat', monkeys{imonkey}));
            % returns S(igrat).trial(itrial).spikes
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                S(igrat).trial(itrial).counts = bin_spikes(S(igrat).trial(itrial).spikes, binWidth);
            end
        end
        
        save(sprintf('./S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end
    
    
%%  Compute trial-averaged population activity (PSTHs)

    for imonkey = 1:length(monkeys)
        fprintf('computing PSTHs for %s\n', monkeys{imonkey});
        
        load(sprintf('./S_%s.mat', monkeys{imonkey}));
            % returns S(igrat).trial(itrial).spikes
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        for igrat = 1:num_grats
            mean_FRs = zeros(size(S(igrat).trial(1).counts));
            for itrial = 1:num_trials
                mean_FRs = mean_FRs + S(igrat).trial(itrial).counts;
            end
            S(igrat).mean_FRs = mean_FRs / num_trials;
        end
        
        save(sprintf('./S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end
    

        

