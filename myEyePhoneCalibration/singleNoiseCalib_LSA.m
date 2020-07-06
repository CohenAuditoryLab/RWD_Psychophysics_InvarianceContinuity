function [testLevel] = singleNoiseCalib_LSA(amp,scaleFactor,invFilter)

Fs = 44100;
tRamp = 0:1/Fs:0.1-1/Fs;
rUp = (0:1:length(tRamp))/length(tRamp);
paddingZone = 6*length(rUp);

noiseDur = 500;
noise = randn(Fs*noiseDur/1000, 1);
noise = conv(noise,invFilter) * amp;

    
noise_padded = [zeros(1,paddingZone) noise' zeros(1,paddingZone)];

recObj = audiorecorder(Fs,24,1,0);
%player = audioplayer(noise_padded,Fs,24);

% player = dotsPlayableWave();
% player.sampleFrequency = Fs;
% player.intensity = 1;
% player.wave = noise_padded;
% player.prepareToPlay;

record(recObj)
%player.play
% playblocking(player)
sound(noise_padded,Fs,24);
pause(length(noise_padded)/Fs)
%player.stop;
stop(recObj)

noiseC = getaudiodata(recObj);

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
mean3 = mean(filteredData(end-round(length(filteredData)/3):end));
std3 = std(filteredData(end-round(length(filteredData)/3):end));

avgMean = (mean1+mean3)/2;
avgStd = sqrt((std1^2+std3^2)/2);

thresh = avgMean+2*avgStd;
%%% find where signal crosses thresholds (beginning and end)
start = find(filteredData(paddingZone:end)>=thresh | filteredData(paddingZone:end)<= -thresh,round(0.05*Fs));
finish = find(filteredData(1:end-paddingZone)>=thresh | filteredData(1:end-paddingZone)<= -thresh,round(0.05*Fs),'last');

noiseC = noiseC(paddingZone-1+start(1):finish(end));

subplot(2,2,4)
plot(noiseC)

% Francisco
MicGain = 1;
Gain = 10^(MicGain/20); 
MicSensitivity = 1000;
Sensitivity=MicSensitivity/1000;
Po=2.2E-5;
testLevel=20*log10(std(noiseC)/Gain/Sensitivity/Po);

% Adam
% noiseC = (noiseC-mean(noiseC))*scaleFactor;
% testLevel = 20*log10(sqrt(mean(noiseC.^2))/Po); % scale factor: 1 V/Pa


