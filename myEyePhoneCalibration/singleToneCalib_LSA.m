function [data, Fs] = singleToneCalib_LSA(freq,amp)

Fs = 44100;

% % Adam's original code (until the line 'tone = ...')
tRamp = 0:1/Fs:0.1-1/Fs;
rUp = (0:1:length(tRamp))/length(tRamp);
% rDown = fliplr(rUp);
% rWindow = amp*[rUp ones(1,length(0:1/Fs:0.5-1/Fs)-2*length(rUp)) rDown];

% time = 0:1/Fs:0.5-1/Fs;
% tone = rWindow.*sin(2*pi*freq*time);

% Lalitta: use my own stim generator (next 2 lines)
toneDur = 500;
[~,tone] = stimGen_noise_embedded_HL(freq,freq,'L',toneDur,100,amp,amp,amp,0,Fs);

paddingZone = 6*length(rUp);
tone = [zeros(1,paddingZone) tone zeros(1,paddingZone)];


recObj = audiorecorder(Fs,24,1,2);
% player = audioplayer(tone,Fs,24);

player = dotsPlayableWave();
player.sampleFrequency = Fs;
player.intensity = 1;
player.wave = tone;
player.prepareToPlay;

record(recObj)
player.play
% playblocking(player)
% sound(tone,Fs,24);
pause(length(tone)/Fs)
player.stop;
stop(recObj)

data = getaudiodata(recObj);

d = fdesign.bandpass('N,F3dB1,F3dB2',10,200,20000,Fs);
Hd = design(d,'butter');

% fvtool(Hd)
filteredData = filter(Hd,data);

figure(1)
subplot(2,2,1)
plot(tone)
subplot(2,2,2)
plot(data)
subplot(2,2,3)
plot(filteredData)

% % Adam's original code
% deriv = zeros(1,length(data)-1);
% for j=1:length(deriv)
%     deriv(j) = (data(j+1)-data(j))/(1/Fs);
% end
% 
% mean1 = mean(deriv(floor(0.1*Fs):floor(0.35*Fs)));
% std1 = std(deriv(floor(0.1*Fs):floor(0.35*Fs)));
% mean3 = mean(deriv(end-round(length(deriv)/3):end));
% std3 = std(deriv(end-round(length(deriv)/3):end));
% 
% avgMean = (mean1+mean3)/2;
% avgStd = sqrt((std1^2+std3^2)/2);
% 
% thresh = avgMean+3.1*avgStd;

% %%% find where signal crosses thresholds (beginning and end)
% start = find(deriv>=thresh,round(0.05*Fs));
% finish = find(deriv(1:end-round(length(deriv)/3))>=thresh,round(0.05*Fs),'last');

% Lalitta
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

data = data(paddingZone-1+start(1):finish(end));

subplot(2,2,4)
plot(data)

disp('hey')