function choice = race_trial(start_point, thresh, dt, sigma, bias)

    i = 1;
    
    % W of choices
    meanOfW_choice1 = 0;
    stdOfW_choice1 = sqrt(dt(1));
    meanOfW_choice2 = 0;
    stdOfW_choice2 = sqrt(dt(2));
    
    % initialization of the choices
    X_choice1 = start_point;
    X_choice2 = start_point;
    
    % drift diff
    while (X_choice1(i) < (thresh(1)) && X_choice2(i) > thresh(2))
        dW_choice1 = normrnd(meanOfW_choice1,stdOfW_choice1);
        dW_choice2 = normrnd(meanOfW_choice2,stdOfW_choice2);
        
        dx_choice1 = bias(1)*dt(1) + sigma(1)*dW_choice1;
        dx_choice2 = bias(2)*dt(2) + sigma(2)*dW_choice2;
        
        X_choice1(i+1) = X_choice1(i) + dx_choice1;
        X_choice2(i+1) = X_choice2(i) + dx_choice2;
        
        i = i+1;
    end
    
    if(X_choice1(end) >= abs(thresh(1)))
        choice = 1;
        X_choice1(end)  = thresh(1);
    elseif(X_choice2(end) <= abs(thresh(2)))
        choice = -1;
        X_choice2(end)  = thresh(2);
    end
    
    
    figure;
    hold on;
    plot(1:i, X_choice1,'LineWidth',1.5);
    plot(1:i, X_choice2,'LineWidth',1.5);
    yline(thresh(1),'LineWidth',1.5,'Color','r')
    yline(thresh(2),'LineWidth',1.5,'Color','g')
    ylim([thresh(2)-1 thresh(1)+1]);
    hold off
    legend('Evidence X1','Evidence X2','choice = 1','choice = -1','Location','best')
    grid on; grid minor;
    title("Evidence of Two Choices, Choice = " + choice,'interpreter','latex');
    xlabel('Sample','interpreter','latex');
    ylabel('Evidence Value(t)','interpreter','latex');

end