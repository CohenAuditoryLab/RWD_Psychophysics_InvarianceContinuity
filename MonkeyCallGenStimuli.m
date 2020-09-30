%% Code for generating monkey call stimuli for SFA
%%% 2020-01-17 started code
%%%
%%%


%% Read in WAV files

folder_location = 'C:\Users\ronwd\.spyder-py3\SFA_PostCOSYNEAPP-master\Monkey_Calls\HauserVocalizations\Monkey_Wav_50K\Wav\';

%Just hard coding for now
voc1_name = 'co1405.wav';
voc2_name = 'ha2unk.wav';

[coo1, Fs1] = audioread([folder_location voc1_name]);
[grunt1, Fs2] = audioread([folder_location voc2_name]);

%Check Fs
match = Fs1==Fs2

%set up something for Fs mismatch
%update 2020-01-21:
%Got database from Yale that has all of the vocalization resampled up to
%50kHz


%% Code for amplitude modulation
%Not Really working easily with the monkey calls because they are so
%short...May need to think of a different effect
%Not really working, most of the time applying the AM means there is just
%no sound.

% %Message signal parameters
% alpha = .5; %modulation index, unitless, (0 to 1) where 0 is no modulation
% Mess_freq_distribution = 1:.5:10; %apparently <20pi or <10 Hz is tremolo effect range (check this)
% %update: checked at perceptually seems true, at least 20 or more is
% %definitely not just tremolo any more, and 15 is getting there
% 
% coo_AM = cell(length(Mess_freq_distribution),1); %set up cell structure like in other blocks
% grunt_AM = cell(length(Mess_freq_distribution),1); %set up cell structure like in other blocks
% 
% 
% index = 1; %pick which of the base stimuli you want to do the modulation on
% 
% phim = 0; %phase shift phi, units of these are seconds for now, keeping zero for now
% 
% dt1 = 1/Fs1;
% dt2 = 1/Fs2;
% 
% Full_Duration1 = length(coo1)/Fs1;
% Full_Duration2 = length(grunt1)/Fs2;
% 
% t1=0:dt1:Full_Duration1; %Message signal should be as long as stimulus
% t2=0:dt2:Full_Duration2;
% 
% 
% for i = 1:length(Mess_freq_distribution)
% 
% Fm = Mess_freq_distribution(i); %Pull in a new frequency each time    
%     
% %Create message signal, right now sine wave just because it is easy
% %note message comes from idea behind AM radio, tune to carrier but modulations on carrier hold the actual information
% %make sure they are the same length as the stimulus
% 
% Message1 = sin(2*pi*Fm*(t1+phim)); Message1 = Message1(1:length(coo1)); 
% Message2 = sin(2*pi*Fm*(t2+phim)); Message2 = Message2(1:length(grunt1));
% 
% 
% coo_AM{i} = coo1 + Message1*alpha*coo1;
% 
% grunt_AM{i} = grunt1 + Message2*alpha*grunt1;
% 
% %audiowrite([folder_location '\Set1_AM\' coo_name '_AM_' num2str(Fm) '.wav'],coo_AM{i},Fs1)
% %audiowrite([folder_location '\Set1_AM\' grunt_name '_AM_' num2str(Fm) '.wav',grunt_AM{i},Fs2])
% 
% 
% 
% e







%% Clutter
% 
%2020-01-21 trying a preselected coo and grunt for now and going to make
%the clutter out of coos and grunts as well.  Will check this result then
%maybe upscale everything.  Specific coo and grunt selected is above.

%For first pass also going to hard code clutter.  Maybe either want to sort
%calls by type into separate folders or will have to do a random draw from
%dir of the vocalization folder.

%Also may want to set up the poisson noise idea as well...need to think on
%this.  For now just adjust this code.

%Today 2020-02-18 focus is on getting the stimuli library generated for
%SFA.  Then tomorrow/this afternoon will be trying to get the SFA code
%setup properly and then running that over on kilosort.  While this is
%running put together presentation for journal club.

%2020-02-18 note the below is less useful now since we are making a large
%batch of stimuli but keep it for now and just note that we will be using
%the Feb 18 2020 file moving forward.  Also will be moot if we get this
%done by the end of the day.


Save_Folder = 'SFA_Monkey_Clutter_Stimuli';

%Make a unique folder for each day
Stimulus_Folder = [date '_Stimuli']; 

if ~isfolder([cd '\' Save_Folder '\' Stimulus_Folder])
    
    mkdir([cd '\' Save_Folder '\' Stimulus_Folder])
    
end

Outer_Path = [cd '\' Save_Folder '\' Stimulus_Folder];


%% Set Clutter "vocabulary"
%Adjusting this to run as a loop over all pairs of vocalizations...
%set up random sampling procedure now. 

%Do some scratch work on what the axes for this should be:
%obviously have snr as one dimension
%For now I think just do pairs, but we will scale up to number of
%vocalizations/potentially ask Yale again.

all_vocal_files = dir(folder_location);
all_vocal_files = all_vocal_files(3:end);
all_vocal_files = {all_vocal_files(:).name};%cell array of all file names

%remove the files from above from the potential clutter files.
Signal_files={voc1_name,voc2_name};
Clutter_files = all_vocal_files;
shared_inds = ismember(Clutter_files,Signal_files);
Clutter_files(shared_inds)= []; %set signal files to [] (i.e. remove from cell array) so they can't be called



%% Put clutter vocalizations into cell array

Stim_Clutter = cell(length(Clutter_files),1);

for i =1:length(Clutter_files)
  
    [Full_stim ,Fs] = audioread([folder_location Clutter_files{i}]);
    Full_Duration = length(Full_stim)/Fs;
    
    Stim_Clutter{i}.Full_stim = Full_stim;
    Stim_Clutter{i}.Fs = Fs;
    Stim_Clutter{i}.Full_Duration = Full_Duration;
    
    
end
%% Trying new thing, going to circle shift the vocalizations to really ruin their structure and see what happens 2020-01-28
meanshift = 5000;
shiftvar = meanshift*2;


for i = 1:length(Clutter_files)
    
    shifter = round(normrnd(meanshift,shiftvar));
    
    Stim_Clutter{i}.Full_stim = circshift(Stim_Clutter{i}.Full_stim,shifter(1));
    
end

'done'

%% Make clutter chorus
[ychorus, randomgrabs]=make_chorus_calls_RWD(Stim_Clutter, Outer_Path); %function to create choruses of calls, Outer_Path is folder to save file in.


%% Invar full monty code, explore later
