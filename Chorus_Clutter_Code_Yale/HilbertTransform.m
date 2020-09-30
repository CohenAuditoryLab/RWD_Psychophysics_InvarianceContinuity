% try envelope detection with hilbert transform

% define a time vector
t = [0:0.01:9.99];

% our signal - a sine wave with a time varying amplitude
x = sin(4*pi*t); % the sine wave
envelope = 5+0.5*sin(pi*t); % the time varying amplitude
x_env = x.*envelope; % the complete signal

% add some random noise to make it look a bit
% more like measured data
signal = x_env + 0.2*rand(1,1000);

% find the hilbert transform of the measured signal
signal_h = hilbert(signal);

% square the hilbert transform and take the
% +ve square root to get the estimated envelope
envelope_est = sqrt(signal_h.*conj(signal_h));

% plot the original signal, the true envelope
% and the estimated envelope
plot(t,signal,t,envelope,t,envelope_est);

% find the mean of the true envelope
mean_true = mean(envelope)

% find the mean of the estimated envelope
mean_est = mean(envelope_est)