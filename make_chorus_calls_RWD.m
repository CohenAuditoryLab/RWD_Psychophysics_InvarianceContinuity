function [ychorus2, randomgrabs]= make_chorus_calls_RWD(Stim_Clutter,clutter_dir,pairnum)
% Make a chorus from the songs of N different ZFs
% Assumes that the stimuli are all saved with the same sampling rate
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

%2019-10-17 need to add input parser or more option arguments, for now just
%leaving hard coding in for the function

%Stim_Clutter is the cell array that hold on the info regarding the clutter
%stimuli
%clutter_dir is the full path the clutter wav files at a string

%ychorus is the matrix of values for the chorus that was created
%randomgrabs are the specific vocalization that were grabbed from the
%battery
%currently some issues with saving.  Need to manual change save name if
%want multiple chorus for a given battery.


%% Set up some basics
%2020-07-20 doubled number of vocalizations to combat gaps in the choruses

nsongs = 20; % # of vocalizations comprising the chorus
%2019-10-17: lets start by trying to use 10 calls
%2020-01-21 keeping this for now, reduced mean shift to 5000 inds since
%monkey calls are so much shorter than the artifical stimulus

meanshift = 5000; %set up mean for normal distributions of shifts
shiftvar = meanshift; %set up variance for normal distribution of shifts
filtlength = 500; % length of the filter for computing envelope, in samples 2020-01-21 double check if this works correctly for calls.  Seems to work correctly



outName = [num2str(pairnum) '_ClutterChorus_',num2str(nsongs),'calls']; % Name of resultant wave file
%stimDir = clutter_dir; % Directory containing wave files
saveDir = clutter_dir; % Save Directory

%% Choose and load the wav files

fs = Stim_Clutter{1}.Fs;
nbits = 24;

%unclear if we need the doubling but will leave it for now for ease of
%implementation

ysong = cell(nsongs,1);
randomgrabs = randsample(length(Stim_Clutter),nsongs); %to allow any of the clutter stimuli to be selected, probably should save this out though
%% Need to think on and update this 

%2020-01-21 do we want to shift the monkey vocalizations around?  Probably
%right?
%I think we want this to work more like random noise block where
%vocalizations can start at random points in block rather then circle
%shifting...i.e. chattering away.  This would make each ysong the same
%length as well which would help with the issues below

chorus_duration = round(3 *fs); %set the length of the chorus during which any vocalization can appear.

for i = 1:nsongs
   
  %2020-07-20 To increase overlap may set this so first half of songs have to go into
  %first half of chorus and second half of songs have to go into second
  %half .  Also probably will add check of some kind of how much of the
  %resulting chorus is silence (sum down the columns (i.e. across spectra)
  %of spectrogram and see if there are pockets of zeros.
  %Related to above could do a running thing where there is forced overlap
  %the whole time, but this seems a bit artifical so going to leave out for
  %now and see what happens
  
  ysong_Temp = zeros(chorus_duration,1); %set blank template for each y song
  edge_ind = chorus_duration - length(Stim_Clutter{randomgrabs(i)}.Full_stim) - 1; %last index the call can be placed into and the full call heard
  starting_ind = randsample(edge_ind,1); %choose where call starts
  ysong_Temp(starting_ind:starting_ind+length(Stim_Clutter{randomgrabs(i)}.Full_stim)-1) = Stim_Clutter{randomgrabs(i)}.Full_stim;
  %Stim_Clutter{randomgrabs(i)}.Full_stim;
  %shifter = round(normrnd(meanshift,shiftvar)); %randperm(maxshift); %might want to change how this is implemented so generally get bigger shifts rather than allow for really small shifts still
  %shiftDir = sign(rand-0.5); %direction of the phase shift, 2019-10-17
  %don't need unless doing randperm implentation as normal implementation
  %can take on negative values.
  
  %Rounding in above is necessary snice circshift needs interger inputs
  %ysong{i} = circshift(ysong_Temp,shifter);%o.g also concated circshift([ysong_Temp; ysong_Temp],shiftDir*shifter(1));
  ysong{i} = ysong_Temp; %simply place the template in the cell position
  
  %clear ysong_Temp
    
end

%% Make the chorus (need to change!)
%Update 2020-01-21: now all files are the same length so below should not
%be an issue.

% Find the shortest wav file
mindur = inf;
for i = 1:nsongs
    mindur = min(mindur,length(ysong{i}));
end

% Shorten all of the songs to the same length and put them into a matrix
for i = 1:nsongs
    songmatrix(i,:) = [ysong{i}(1:mindur)];
end

ychorus = [];
numofblocks = 25;
for i =1:numofblocks
% Multiply each song by a random weighting (these are all pretty close to 1)
w = rand(nsongs,1);
w = w/4;
w = w + 1 -1/8;
songmatrixalt = repmat(w,1,size(songmatrix,2)).*songmatrix;

% Get the resulting chorus
ychorus =[ychorus mean(songmatrixalt)]; %horizontal concate all the different random weight noises

end

%Update 2019-10-17 To make the noise file longer and more disordered, could
%concate circshifted copies of ychorus or generate ychorus multiple times
%and then circshift the concatenated copies
%For the sake of not having any structure I am just going to make a giant
%noise file even though really would need to have this scale according to
%the number of training examples given.  Currently that number is 5, but I
%will make this long enough to do 25.  I.e. let's make 25 ychoruses and
%concatenate them.  Could circshift as well but I am not sure that is super
%necessary
%alternativly could also generate new songmatrix by changing randomly
%grabbed stimuli but that may be too much change...just leave this alone
%for now/implement it and then come back and fix after chatting with Yale

%% Regularize the energy over time to remove peaks and troughs
% Get the normalized envelope of the chorus
env = abs(ychorus);
filtt = ones(filtlength,1)/filtlength;
envelope = convn(env,filtt','same');
envelope2 = envelope/max(envelope);

% Regularize the engergy of the chorus over time
ychorus2 = ychorus.*(1./envelope2);

%2020-07-20 check for pockets of silence in ychorus

figure;
spectrogram(ychorus2,[],[],[],fs,'yaxis')
testing = sum(ychorus2,1);
plot(testing)

% Get the envelope of the regularized chorus to make sure that it's flat
clear env filtt
env = abs(ychorus2);
filtt = ones(filtlength,1)/filtlength;
envelope3 = convn(env,filtt','same');

%% Plot the non-regularized and regularized chorus'
% figure
% subplot(2,1,1)
% plot(ychorus,'k-')
% hold on
% plot(envelope,'r-')
% title('original chorus')
% legend('Chorus','Envelope','Location','NE')
% 
% subplot(2,1,2)
% plot(ychorus2,'k-')
% hold on
% plot(envelope3,'r-')
% title(['regularized chorus with filter = ',num2str(filtlength),' samples'])

%% Save the chorus
cd(saveDir)

audiowrite([outName '.wav'],ychorus2,fs,'BitsPerSample',nbits)

disp('done')