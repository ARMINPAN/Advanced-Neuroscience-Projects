function probability = policyUpdate(currentState, value_map, policy)   
    beta = 10;
    probability = policy(currentState(1),currentState(2),:);
    
    value_map = [(-10^10000)*ones(1,15); value_map; (-10^10000)*ones(1,15)];
    value_map = [(-10^10000)*ones(17,1), value_map, (-10^10000)*ones(17,1)];
    
    currentState = currentState + 1;
    allNeighborsProb = exp(beta*value_map(currentState(1),currentState(2)-1))+...
       exp(beta*value_map(currentState(1)-1,currentState(2))) + ...
        exp(beta*value_map(currentState(1),currentState(2)+1)) + ...
        exp(beta*value_map(currentState(1)+1,currentState(2)));
    
    sum_of_values = value_map(currentState(1),currentState(2)-1) + ...
        value_map(currentState(1)-1,currentState(2)) + ...
        value_map(currentState(1),currentState(2)+1) + ...
        value_map(currentState(1)+1,currentState(2));
    
    if(sum_of_values ~= 0)
        % left prob 
        probability(1) = exp(beta*(value_map(currentState(1),currentState(2)-1)))/...
            allNeighborsProb;

        % up prob 
        probability(2) = exp(beta*(value_map(currentState(1)-1,currentState(2))))/...
            allNeighborsProb;

        % right prob 
        probability(3) = exp(beta*(value_map(currentState(1),currentState(2)+1)))/...
            allNeighborsProb;

        % bottom prob 
        probability(4) = exp(beta*(value_map(currentState(1)+1,currentState(2))))...
            /allNeighborsProb;
    end
end