function [] = singleNoiseCalibFilterGen_LSA(calibNoiseFileName)
Fs = 44100;
tRamp = 0:1/Fs:0.1-1/Fs;
rUp = (0:1:length(tRamp))/length(tRamp);
paddingZone = 5*length(rUp);

% Computing RMS SPL - Francisco
NFFT = 1024*4;
ATT = 80;
L = 395;
MicGain = 1;
MicSensitivity = 1000;
Gain = 10^(MicGain/20);               %Microphone Amplifier Gain

noiseDur = 10000;
noise = randn(Fs*noiseDur/1000, 1);

noise_padded = [zeros(1,paddingZone) noise' zeros(1,paddingZone)];

recObj = audiorecorder(Fs,24,1,0);
% player = audioplayer(noise_padded,Fs,24);

% player = dotsPlayableWave();
% player.sampleFrequency = Fs;
% player.intensity = 1;
% player.wave = noise_padded;
% player.prepareToPlay;


record(recObj)
%player.play
%playblocking(player)
 sound(noise_padded,Fs,24);
pause(length(noise_padded)/Fs)
%player.stop;
stop(recObj)

noiseC = getaudiodata(recObj);

figure(1)
subplot(2,2,1)
plot(noise)
subplot(2,2,2)
plot(noiseC)

d = fdesign.bandpass('N,F3dB1,F3dB2',10,50,20000,Fs);
Hd = design(d,'butter');

% fvtool(Hd)
filteredData = filter(Hd,noiseC);

subplot(2,2,3)
plot(filteredData)

mean1 = mean(filteredData(floor(0.1*Fs):floor(0.35*Fs)));
std1 = std(filteredData(floor(0.1*Fs):floor(0.35*Fs)));
mean3 = mean(filteredData(end-paddingZone:end));
std3 = std(filteredData(end-paddingZone:end));

avgMean = (mean1+mean3)/2;
avgStd = sqrt((std1^2+std3^2)/2);

thresh = avgMean+2*avgStd;
%%% find where signal crosses thresholds (beginning and end)
start = find(filteredData(paddingZone:end)>=thresh | filteredData(paddingZone:end)<= -thresh,round(0.05*Fs));
finish = find(filteredData(1:end-paddingZone)>=thresh | filteredData(1:end-paddingZone)<= -thresh,round(0.05*Fs),'last');

noiseC = noiseC(paddingZone-1+start(1):finish(end));

subplot(2,2,4)
plot(noiseC)

%% from Francisco

%Converting Signals to Pascals, note X does is in Volts and does not need
%to be converted to Pascals. Only Y is converted.
X = noise(end-length(noiseC)+1:end);
X = X - mean(X);
Y = noiseC.*1000./Gain/MicSensitivity;
Y = Y - mean(Y);

%Generating Kaiser Window for Spectrum Calculations
[Beta,N,wc] = fdesignk(ATT,0.1*pi,pi/2);
W = kaiser(NFFT,Beta);

%Generating Speaker Transfer Functions
[P,F]= spectrum(Y,X,NFFT,NFFT*7/8,W,Fs);
Pyy = P(:,1);
Pxx = P(:,2);
Pyx = P(:,3);
H = Pyx./Pxx;  %Forward transfer function, NOT the inverse, FREQZ inverts!

%Plotting Input Time Data
figure(2)
subplot(2,2,1)
noise = noise(end-length(noiseC)+1:end);
plot((0:length(noise)-1)/Fs,noise,'k')

%Plotting Measured Micropone Response
subplot(2,2,3)
hold off
xlim([0 10])
ylabel('Volts')
plot((0:length(noise)-1)/Fs,noiseC,'k')
hold on

title('Microphone Measurement')
xlim([0 10])
ylim([min(noiseC)  max(noiseC)])
xlabel('Time (sec)')
ylabel('Volts')
pause(0.1)

%Spectrum Figure
subplot(2,2,[2 4])
hold off
Po=2.2E-5;
Offset=10*log10((872161/Fs).^2) ;

plot(F/1000,10*log10(Pyy/NFFT*2./Po.^2),'k')
hold on
plot(F/1000,10*log10(Pyx.^2/NFFT*2./Po.^2)+Offset,'r-.')
plot(F/1000,10*log10(Pyx.^2/NFFT*2./Po.^2),'color',[0.5 .5 .5])
%     if AcquireSelect==1   %Validation Mode
%         plot(F/1000,10*log10(abs(Pyx).^2./NFFT*2./Po.^2),'g');
%     end
ylabel('SPL (dB)')
xlim([0 30])
xlabel('Frequency (KHz)')
%mtit('Calibration')

%Generates the Inverse Filter Impulse Response
W = (0:NFFT/2)/(NFFT/2)*pi;
[Hband] = bandpass(0,20000,1500,Fs,ATT,'n');
[B,A] = invfreqz(H,W,0,L-length(Hband));
CalibData.hinv=conv(A,Hband);
CalibData.Gain=8/max(abs(conv(CalibData.hinv,noise)));
CalibData.hinv=CalibData.Gain*CalibData.hinv;     %Assures that output amplitude does not exceed +/- 8 Volts

%Storing Transfer Function Results
CalibData.H=H;
CalibData.Hinv=1./H;
CalibData.Pyy=Pyy;
CalibData.Pxx=Pxx;
CalibData.Pyx=Pyx;

save(calibNoiseFileName,'CalibData');

%% validate
noiseDur = 10000;
noise = randn(Fs*noiseDur/1000, 1);
noise = conv(CalibData.hinv,noise);

noise_padded = [zeros(1,paddingZone) noise' zeros(1,paddingZone)];

recObj = audiorecorder(Fs,24,1,0);
% player = audioplayer(noise_padded,Fs,24);

%player = dotsPlayableWave();
%player.sampleFrequency = Fs;
%player.intensity = 1;
%player.wave = noise_padded;
%player.prepareToPlay;


record(recObj)
%player.play
% playblocking(player)
sound(noise_padded,Fs,24);
pause(length(noise_padded)/Fs)
%player.stop;
stop(recObj)

noiseC = getaudiodata(recObj);

figure(3)
subplot(2,2,1)
plot(noise)
subplot(2,2,2)
plot(noiseC)


d = fdesign.bandpass('N,F3dB1,F3dB2',10,50,20000,Fs);
Hd = design(d,'butter');

% fvtool(Hd)
filteredData = filter(Hd,noiseC);

subplot(2,2,3)
plot(filteredData)

mean1 = mean(filteredData(floor(0.1*Fs):floor(0.35*Fs)));
std1 = std(filteredData(floor(0.1*Fs):floor(0.35*Fs)));
mean3 = mean(filteredData(end-paddingZone:end));
std3 = std(filteredData(end-paddingZone:end));

avgMean = (mean1+mean3)/2;
avgStd = sqrt((std1^2+std3^2)/2);

thresh = avgMean+2*avgStd;
%%% find where signal crosses thresholds (beginning and end)
start = find(filteredData(paddingZone:end)>=thresh | filteredData(paddingZone:end)<= -thresh,round(0.05*Fs));
finish = find(filteredData(1:end-paddingZone)>=thresh | filteredData(1:end-paddingZone)<= -thresh,round(0.05*Fs),'last');

noiseC = noiseC(paddingZone-1+start(1):finish(end));

subplot(2,2,4)
plot(noiseC)

noise = noise(end-length(noiseC)+1:end);
X = noise;
X = X - mean(X);
Y = noiseC.*1000./Gain/MicSensitivity;
Y = Y - mean(Y);

%Generating Kaiser Window for Spectrum Calculations
[Beta,N,wc] = fdesignk(ATT,0.1*pi,pi/2);
W = kaiser(NFFT,Beta);

%Generating Speaker Transfer Functions
[P,F]= spectrum(Y,X,NFFT,NFFT*7/8,W,Fs);
Pyy = P(:,1);
Pxx = P(:,2);
Pyx = P(:,3);

%Plotting Input Time Data
figure(4)
subplot(2,2,1)
plot((0:length(noise)-1)/Fs,noise,'k')

%Plotting Measured Micropone Response
subplot(2,2,3)
hold off
xlim([0 10])
ylabel('Volts')
plot((0:length(noise)-1)/Fs,noiseC,'k')
hold on

title('Microphone Measurement')
xlim([0 10])
ylim([min(noiseC)  max(noiseC)])
xlabel('Time (sec)')
ylabel('Volts')
pause(0.1)

%Spectrum Figure
subplot(2,2,[2 4])
hold off
Po=2.2E-5;
Offset=10*log10((872161/Fs).^2) ;

plot(F/1000,10*log10(Pyy/NFFT*2./Po.^2),'k')
hold on
plot(F/1000,10*log10(Pyx.^2/NFFT*2./Po.^2)+Offset,'r-.')
plot(F/1000,10*log10(Pyx.^2/NFFT*2./Po.^2),'color',[0.5 .5 .5])
%     if AcquireSelect==1   %Validation Mode
%         plot(F/1000,10*log10(abs(Pyx).^2./NFFT*2./Po.^2),'g');
%     end
ylabel('SPL (dB)')
xlim([0 30])
xlabel('Frequency (KHz)')
%mtit('Validation')
    

