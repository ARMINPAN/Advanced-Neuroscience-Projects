function plotRawSigPinkNoise(f,pinkNoise,rawSignal,dePinkedSignal,...
    targetElectrode)
    % plot the target signal fft
    figure;
    subplot(1,2,1);
    hold on;
    plot(f,(rawSignal),'LineWidth',2);
    title("Mean over All Trials - Electrode = " + targetElectrode ...
        ,'interpreter','latex')
    xlabel('Log10 Frequency(Hz)','interpreter','latex')
    ylabel('Log10 Power','interpreter','latex') 
    
    % plot pink noise
    hold on;
    plot(f,pinkNoise,'LineWidth',2);
    
    % plot dePinkNoised data log log
    plot(f,(dePinkedSignal),'LineWidth',2);
    legend('Raw Signal','Pink Noise','Pink Noise Free');
    grid on; grid minor;

    % plot dePinkNoised data 
    subplot(1,2,2);
    % *10 to make it db - its just log10 at first
    plot(10.^f,(10.^dePinkedSignal),'LineWidth',2);
    title("Mean over All Trials - Pink Noise Free - Electrode = " + targetElectrode  ...
        ,'interpreter','latex')
    xlabel('Frequency(Hz)','interpreter','latex')
    ylabel('Power','interpreter','latex') 
    grid on; grid minor;
    
end