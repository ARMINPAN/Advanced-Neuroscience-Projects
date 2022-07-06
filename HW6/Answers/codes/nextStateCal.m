function nextState = nextStateCal(currentState, policy)

    x = rand;
    p = squeeze(policy(currentState(1)+1,currentState(2)+1,:));
    
    nextState = currentState;
    if(0<=x && x<=sum(p(1))) % left
        nextState = currentState + [0 -1];
    elseif(sum(p(1))+0.000000000000001<=x && x<= sum(p(1:2))) % up
        nextState = currentState + [-1 0];
    elseif(sum(p(1:2))+0.0000000000001<=x && x<= sum(p(1:3))) % right
        nextState = currentState + [0 1];
    elseif(sum(p(1:3))+0.0000000000001<=x && x<= sum(p(1:4))) % down
        nextState = currentState + [1 0];
    end
end