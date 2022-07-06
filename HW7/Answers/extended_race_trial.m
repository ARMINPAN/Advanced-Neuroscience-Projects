function choice = extended_race_trial(start_point, thresh, dt, sigma, bias, time_interval)

    % W of choices
    meanOfW_choice1 = 0;
    stdOfW_choice1 = sqrt(dt(1));
    meanOfW_choice2 = 0;
    stdOfW_choice2 = sqrt(dt(2));
    
    % initialization of the choices
    X_choice1 = start_point;
    X_choice2 = start_point;
    
    % drift diff
    for i = 1:length(time_interval)-1
        dW_choice1 = normrnd(meanOfW_choice1,stdOfW_choice1);
        dW_choice2 = normrnd(meanOfW_choice2,stdOfW_choice2);
        
        dx_choice1 = bias(1)*dt(1) + sigma(1)*dW_choice1;
        dx_choice2 = bias(2)*dt(2) + sigma(2)*dW_choice2;
        
        X_choice1(i+1) = X_choice1(i) + dx_choice1;
        X_choice2(i+1) = X_choice2(i) + dx_choice2;
    end
    
    if(abs(X_choice1(end)) > abs(X_choice2(end)))
        choice = 1;
    elseif(abs(X_choice1(end)) < abs(X_choice2(end)))
        choice = -1;
    else
        choice = randi(2,1);
        if(choice == 2)
            choice = 1;
        else
            choice = -1;
        end
    end
    
    
    figure;
    hold on;
    plot(time_interval, X_choice1,'LineWidth',1.5);
    plot(time_interval, X_choice2,'LineWidth',1.5);
    yline(thresh(1),'LineWidth',1.5,'Color','r')
    yline(thresh(2),'LineWidth',1.5,'Color','g')
    ylim([thresh(2)-1 thresh(1)+1]);
    hold off
    legend('Evidence X1','Evidence X2','choice = 1','choice = -1','Location','best')
    grid on; grid minor;
    title("Evidence of Two Choices, Choice = " + choice,'interpreter','latex');
    xlabel('Time(s)','interpreter','latex');
    ylabel('Evidence Value(t)','interpreter','latex');

end