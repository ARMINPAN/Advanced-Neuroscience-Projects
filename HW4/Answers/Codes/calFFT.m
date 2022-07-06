function [f,P1] = calFFT(X,Fs)
    L = length(X);
    Y = fft(X); 
    P2 = (abs(Y).^2)/L;
    P1 = P2(2:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(1:(L/2))/L;
end