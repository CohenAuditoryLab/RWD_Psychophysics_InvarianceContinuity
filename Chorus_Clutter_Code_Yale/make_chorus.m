function make_chorus
% Make a chorus from the songs of N different ZFs
% Assumes that the stimuli are all saved with the save sampling rate
%
% User should change:
% nsongs: the # of songs comprising the chorus
% maxshift: the maximum phase shift that each song can be shifted
% filtlength: the length of the filter for removing peaks and valleys.
% 500 is a good length for waves sampled around 48828 Hz.
%
% In general, a shorter filter does a better job, but not too short!! The
% envelope in the top figure panel should follow the envelope of the chorus,
% and the envelope in the lower panel should be fairly flat.

%% Set up some basics
nsongs = 7; % # of songs comprising the chorus
maxshift = 1500; % maximum size of phase shift, in samples
filtlength = 500; % length of the filter for computing envelope, in samples

outName = ['ZFchorus_',num2str(nsongs),'songs']; % Name of resultant wave file
stimDir = '/Volumes/lab/Sounds/SpikeBackup/STRF_Standard_Protocol'; % Directory containing wave files
saveDir = '/Volumes/lab/Sounds/SpikeBackup/all_waves/david_degraded_030510'; % Save Directory

%% Choose and load the wave files
cd(stimDir)
for i = 1:nsongs,
    [songfilename{i}, songpathname{i}] = uigetfile('*.wav',['Choose song ',num2str(i),':']);
end

% Load the wave files
% Concatonate each file with itself to double its length
% Randomize the phase relationships
for i = 1:nsongs,
    cd(songpathname{i})
    [ysong_Temp,fs,nbits] = wavread(songfilename{i});
    shifter = randperm(maxshift);
    shiftDir = sign(rand-0.5); %direction of the phase shift
    ysong{i} = circshift([ysong_Temp; ysong_Temp],shiftDir*shifter(1));
    clear ysong_Temp
end

%% Make the chorus
% Find the shortest wave file
mindur = inf;
for i = 1:nsongs,
    mindur = min(mindur,length(ysong{i}));
end

% Shorten all of the songs to the same length and put them into a matrix
for i = 1:nsongs,
    songmatrix(i,:) = [ysong{i}(1:mindur)];
end

% Multiply each song by a random weighting (these are all pretty close to 1)
w = rand(nsongs,1);
w = w/4;
w = w + 1 -1/8;
songmatrix = repmat(w,1,size(songmatrix,2)).*songmatrix;

% Get the resulting chorus
ychorus = mean(songmatrix);

%% Regularize the energy over time to remove peaks and troughs
% Get the normalized envelope of the chorus
env = abs(ychorus);
filtt = ones(filtlength,1)/filtlength;
envelope = convn(env,filtt','same');
envelope2 = envelope/max(envelope);

% Regularize the engergy of the chorus over time
ychorus2 = ychorus.*(1./envelope2);

% Get the envelope of the regularized chorus to make sure that it's flat
clear env filtt
env = abs(ychorus2);
filtt = ones(filtlength,1)/filtlength;
envelope3 = convn(env,filtt','same');

%% Plot the non-regularized and regularized chorus'
figure
subplot(2,1,1)
plot(ychorus,'k-')
hold on
plot(envelope,'r-')
title('original chorus')
legend('Chorus','Envelope','Location','NE')

subplot(2,1,2)
plot(ychorus2,'k-')
hold on
plot(envelope3,'r-')
title(['regularized chorus with filter = ',num2str(filtlength),' samples'])

%% Save the chorus
cd(saveDir)
wavwrite(ychorus2,fs,nbits,outName)

disp('done')