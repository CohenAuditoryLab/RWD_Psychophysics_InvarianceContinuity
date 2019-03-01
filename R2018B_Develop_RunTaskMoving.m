%altered version of v1 run task that now moves the stimulus according to
%HRTF principles and has 100 ms on-off ramps to prevent clicking issues

%2/25/19 Note depends on having R2018B or later!!!!!!!!!!!!!!!!!!!!!!!!!
%Not need to recalibrate headphones for stimuli most likely

%2/26/19 Went with HRTF applied to snaps as wavs for simplicity of
%implementation.  Keep an eye on memory issues with this.

%Also after get all segments to work, need to clean up all this code,
%probably will be a post-exam thing as only going to run this on a max
%N=3-5

%% 1) Make and Save Stimulus Set

rng shuffle %make sure random seed is random

figure(1);
text(0,0.5,'Generating Stimuli and Setting up Task.  Please Wait...','Color','red','FontSize',14);
axis off



%% 2.1) Make base stimuli
%Assumes GenStimulus.m is in the path**************************************

Num_stim = 1;

Stim_set = cell(Num_stim,1);

for i = 1:Num_stim
    i
    
    Stim_set{i} = GenStimulus(1,0); %generates stimulus with either parallel motion (1,0) or contrary motion (0,1)
    
end
%2/25/19 think about whether should do calib filter before or after
%applyign ramp...doing before for now, but really just for convenience

%apply filter to full stimulus for playing on loop,
%apply to pieces when generating snapshots

load('calibNoise')
load('calibrationNoise-08-Nov-2018')

for i = 1:Num_stim
    
    i
    
    %Simply, convolve stimulus with filter calculate from calibration to
    %equalize power across different frequencies then multiply equalized
    %signal by scaleFactor to make sure 75 dB = 75 dB
    
    Stim_set{i}.Full_stim = conv(Stim_set{i}.Full_stim,CalibData.hinv,'same')... same called since don't want to change the length of the vector
        *calibrationFile.scaleFactor;
    
end


%added to make things easier for interacting with below code

Full_stim = Stim_set{1}.Full_stim;
Full_Duration = Stim_set{1}.Duration;

%% Scrambling to come (though can probably just do the same way we did before...

scrambled = -1;  %Set to one to create discontinuous base stimuli



if scrambled >0
    status = 's'
    pull_foldername = 'Trial_2'; %for file name put number trial and initialize of participant
    pull_filename = '26-Feb-2019_Loop1_C_Stim_set.mat'; %put the file name in based on the date.

    load(['C:\Users\ronwd\Desktop\MATLAB\PsychoPhysicsWithYale\Run_task_RWD_v2\Trials\' pull_foldername  '\' pull_filename])
    
    Stim_set{1}.foldername = pull_foldername; %just to keep things matched
    
    %scramble by taking four random sized pieces and shuffling them
    %do this three times to ensure that properly mixed
    
    starting = 1;
    ending = length(Stim_set{1}.Full_stim);
    scramble.Piece1 = Stim_set{1}.Piece1;
    scramble.Piece2 = Stim_set{1}.Piece2;
    scramble.Piece3 = Stim_set{1}.Piece3;
    
    
    for shuffle = 1:3 %shuffle three times
        
        randinds = randperm(ending,3);
        
        randinds = [starting sort(randinds) ending];
        
        %just shuffle back to front since doing it five times will mess
        %everything up anyways.
        
        scramble.Piece1 = [scramble.Piece1(randinds(4):randinds(5)) scramble.Piece1(randinds(3):randinds(4)) scramble.Piece1(randinds(2):randinds(3)) ...
            scramble.Piece1(randinds(1):randinds(2))];
        
        scramble.Piece2 =[scramble.Piece2(randinds(4):randinds(5)) scramble.Piece2(randinds(3):randinds(4)) scramble.Piece2(randinds(2):randinds(3)) ...
            scramble.Piece2(randinds(1):randinds(2))];
        
        scramble.Piece3 =[scramble.Piece3(randinds(4):randinds(5)) scramble.Piece3(randinds(3):randinds(4)) scramble.Piece3(randinds(2):randinds(3)) ...
            scramble.Piece3(randinds(1):randinds(2))];
        
        
        
    end
    
    scramble.together =  scramble.Piece1 + scramble.Piece2 + scramble.Piece3;
    Stim_set{1}.Full_stim = scramble.together;
    Stim_set{1}.Piece1 = scramble.Piece1;
    Stim_set{1}.Piece2 = scramble.Piece2;
    Stim_set{1}.Piece3 = scramble.Piece3;
    
%     Loop_Duration = 5; %in minutes
%     
%     testloop = repmat(scramble,1,(60/Stim_set{1}.Duration)*Loop_Duration);
%     
%     TestStim1 = audioplayer(testloop, Stim_set{1}.Fs, 24);
%     
%     TestStim1.play
    
Full_stim = Stim_set{1}.Full_stim;
Full_Duration = Stim_set{1}.Duration;
  
    
end

%added to make things easier for interacting with below code



%% Setting up loop and removing clicking

Loop_Duration = 1.666; %in minutes (playing for 100 seconds since this avoids weird clicking issue. And plays exactly 1 full left to right, right to left grouping
%will repeat 3 times to get a 5 minute loop (300 seconds = 5 minutes)

ramp_length = 4410*10;%10 * # ms ramp %150 ms seems to definitely work, working back from that. 

AM_s_and_end = [0 linspace(0,1,ramp_length) ones(1,length(Full_stim)-2*ramp_length-2) linspace(1,0,ramp_length) 0];

Full_stim = Full_stim .* AM_s_and_end;

Loop1 = repmat(Full_stim,1,ceil(60*Loop_Duration)/Full_Duration);

%% Save loops of base stimuli

%set up folder for this trial

%(note assumes things about the path, check/change for Tower of Babel
%stuff)

%Get all of the folders in Trials
Temp_Dir = dir([pwd '\Trials']);

Temp_Dir={Temp_Dir.name};
%Subtract that names has a '.' and '..' entry, remainder is number of
%trials

Trialnum = (length(Temp_Dir)-2 +1); 

%just put into a variable to make it easier
ThisTrialFolder = [pwd '\Trials\Trial_' num2str(Trialnum)];

mkdir(ThisTrialFolder)

if scrambled > 0
    
filename = strcat(date,'_Loop1_DC');

else
    
filename =  strcat(date,'_Loop1_C');

end

audiowrite([ThisTrialFolder '\' filename '.wav'], Loop1, Stim_set{1}.Fs,'BitsPerSample',24) %first save the loop

%Set up later for discontinous stimuli
%filename =  strcat(date,'_Loop2_DC');

%Save audio object to play during trial and Stim set in case
%Note: for now always use Stim_set 1 but will need to switch for when using
%discontinous stimulus.

%BaseStim1 = audioplayer(Loop1, Stim_set{1}.Fs, 24);

if scrambled > 0
    
    filename2 = [ strcat(date,'_Loop1_DC_') 'Stim_set'];
   
else
    filename2 = [ strcat(date,'_Loop1_C_') 'Stim_set'];
end

save([ThisTrialFolder '\' filename2], 'Loop1', 'Stim_set')

%Note to self: can still run code while stimuli plays if want to do that

%% Prepare Apply movement to base stimulus

load 'ReferenceHRTF.mat' hrtfData sourcePosition

%fixes formatting issues for below so that hrtf data has axes in right
%places and that the source position data only include az and el

hrtfData = permute(double(hrtfData),[2,3,1]); 

sourcePosition = sourcePosition(:,[1,2]);

%Example on mathworks seems to do things the below way and use a function
%to actively write to the sound card.  Not positive this is the way we want
%to do it.  Also need to think about if snapshops also are from moving
%thing or if they only come from head on...

%for now use that as model for this code and ask Yale on Monday what we
%specifically want to do

loop_length = 60*5; %length of loop in seconds, just hard coding
durationPerPosition = 5; %can changes this as well

num_of_positions = ceil(loop_length/durationPerPosition);
%note that 90 is left and -90 is right also that making sweeps then
%repmating them so that can deal with short durations
angle_speed = 10; %sets speed by setting steps for linspace
az_set = [linspace(90,-90,angle_speed) linspace(-90,90,angle_speed)]; %[90 270]; 
az = repmat(az_set,[1,floor(num_of_positions/length(az_set))])';

el_scaling = 0; %scaling for elevation
desiredPositions = [az  el_scaling*ones(length(az), 1)];

interpolatedIR  = interpolateHRTF(hrtfData,sourcePosition,desiredPositions);

leftIR = squeeze(interpolatedIR(:,1,:));
rightIR = squeeze(interpolatedIR(:,2,:));

    
%set up filters, file to read, and writer to audio card

leftFilter = dsp.FIRFilter('NumeratorSource','Input port');
rightFilter = dsp.FIRFilter('NumeratorSource','Input port');

fileReader = dsp.AudioFileReader([ThisTrialFolder '\' filename '.wav']);
deviceWriter = audioDeviceWriter('SampleRate',fileReader.SampleRate);

%now are ready to play stimulus as it moves.




%% Make snapshot pool
%slight click with snaps, either keep auditory object open to prevent click
%or maybe add an additional on off ramp for these guys

snapshot_duration = 550; %in ms

binsize_inds = floor((snapshot_duration/1000)*Stim_set{1}.Fs); %size of the bins in inds given snap duration

limit_length = (Stim_set{1}.Duration * Stim_set{1}.Fs-binsize_inds)-1; %last safe index given binsize

trials = 100; %number of snaps to generate

save_snaps = cell(trials,1); %save all snap shots for later processing

stimulus_id = 1; %for now

Stim = Stim_set{stimulus_id}; %shorthand for generating below easily

%Notes assumes even number of trials
messed_with_save = [ones(trials/2,1); zeros(trials/2,1)]; %just going to make half altered and half not with random drawing

for i = 1:trials
    
    i
    
    snap.duration = snapshot_duration;
    
    messed_with= messed_with_save(i);
    
    snap.altered = messed_with;


if messed_with == 0
    
    snap.start = randsample(limit_length,1);
    snap.end = snap.start + binsize_inds;
    
    snap.snap_piece = Stim.Full_stim(snap.start:snap.end);
    
    snap.audio = audioplayer(snap.snap_piece,Stim.Fs,24);
    
    
else
   
    snap.start = randsample(limit_length,1);
    snap.end = snap.start + binsize_inds;
  
    %Create with each individual, unfiltered piece so that can apply filter
    %later
    
    snap_piece = Stim.Piece1(snap.start:snap.end)+Stim.Piece3(snap.start:snap.end);%+Stim.Piece2(snap.start:snap.end)
    
    %currently always grabs the second frequency, but this doesn't mean it
    %only changes the middle frequency of the stimulus!  Each piece is
    %grabbed randomly and assigned to a piece randomly.  
    
    %update: 12/11/18 below is unnecessary if simply don't include
    %Stim.Piece2() in above.  Changed accordingly
    %snap_piece = snap_piece - Stim.Piece2(snap.start:snap.end);
    
    %maybe will change later to do a random perturbation, but for now:
    %just have one of the pitches switch direction of sweep
    
    %snap_piece = snap_piece + fliplr(Stim.Piece2(snap.start:snap.end));
    
    %or add in a new frequency by shifting frequency up or down
    
    T = -Stim.Duration/2:1/Stim.Fs:Stim.Duration/2;
    Shift = (0.5-binornd(1,.5))*400; %so shift is random up or down by 200hz
    New_Piece2 = Stim.amp*chirp(T,(Stim.SF(2)*Stim.MC)+Shift,Stim.Duration/2,Stim.SF(2)+Shift,'quadratic');
    
    snap.snap_piece = snap_piece + New_Piece2(snap.start:snap.end);
    
    %As before, apply filter and then scale up to correct volume
    snap.snap_piece = conv(snap.snap_piece,CalibData.hinv,'same')...
        *calibrationFile.scaleFactor;
    
    snap.audio = audioplayer(snap.snap_piece,Stim.Fs,24);
    
    
end

save_snaps{i}= snap;

end

%Get new random order of snap shots to play
%2/26/19 for HRTF addition trying to save a wav of each snap...let's see if this
%makes things explode 


NewOrder = randsample(100,100);
if scrambled >0
filename3 = [ strcat(date,'_Loop1_DC_') 'snapshot_variables'];
else
filename3 = [ strcat(date,'_Loop1_C_') 'snapshot_variables'];
end

save([ThisTrialFolder '\' filename3], 'save_snaps', 'NewOrder')
%Create folder to save wavs in 

mkdir([ThisTrialFolder '\snap_wavs'])

%NOTE: saved in order generated, i.e. , 1-50 is altered 51-100 is unaltered
%This is critical so that we keep the correct answer key

%Below seems to work and not be too slow

for snap_ind = 1:length(save_snaps)
    
    snap_ind
    
    audiowrite([ThisTrialFolder '\snap_wavs\snap_' num2str(snap_ind) '.wav'], save_snaps{snap_ind}.snap_piece, Stim.Fs, 'BitsPerSample', 24)
    
    
end


figure(1);
text(0.3,0.4, '...Stimuli Generated','Color','red','FontSize',14);
axis off

%% Ready Joystick

%Note for all below joystick = gamepad controller

% Define joystick ID
ID = 1;
% Create joystick variable
joy=vrjoystick(ID);

%% Listen to Loop
%2/26/19 troubleshooting a bit, started to hear some clicking or other
%issues with this method with HRTF.  Already checked base stim, doesn't
%seem to be any issues.

%Thinking might see if can apply frame by frame before and then play
%First left to right pass has no clicking, high frequency component starts
%to come out more strongly as going right to left

%No clicking on second left to right pass either

%start to hear clicking on third pass for no apparent reason

clf
figure(1);
text(0.45,0.8, 'Please maximize this window','Color','red','FontSize',14);
axis off

figure(1);
text(0.45,0.6, 'Please set headphone volume to 30','Color','red','FontSize',14);
axis off

pause(8)

figure(1);
text(0.45,0.5, 'Prepare to begin task','Color','red','FontSize',14);
axis off

pause(4)

clf

figure(1);
text(0.45,0.5,'A reference stimulus will play for 5 minutes','Color','red','FontSize',14);
axis off
pause(2)
text(0.45,0.4,'Please listen carefully','Color','red','FontSize',14);
axis off
pause(2)
text(0.45,0.3,'Press any button on the gamepad to begin the reference stimulus','Color','red','FontSize',14);
axis off

while all((button(joy))<1)
    
    pause(0.05)
    
end

pause(1)
clf

figure(1);
text(0.1,0.5,'Playing reference stimulus','Color','red','FontSize',14);
axis off

% Actually play moving stimulus
%note this will play whatever wav file is in filereader,  may need to
%adjust below code for one playing one snapshot

%stuff pulled directly from MATHWORKS example

%As noted elsewhere, repeat this process three times to play clean clip
%since only get 100s of clean playing, (300s = 5 minutes)
%% FIRST PLAY!
leftFilter = dsp.FIRFilter('NumeratorSource','Input port');
rightFilter = dsp.FIRFilter('NumeratorSource','Input port');

fileReader = dsp.AudioFileReader([ThisTrialFolder '\' filename '.wav']);
deviceWriter = audioDeviceWriter('SampleRate',fileReader.SampleRate);


samplesPerPosition = durationPerPosition*fileReader.SampleRate;
samplesPerPosition = samplesPerPosition - rem(samplesPerPosition,fileReader.SamplesPerFrame);

sourcePositionIndex = 1;
samplesRead = 0;

leftChannel = []; %should preallocate fully for speed but for now this is fine
rightChannel = [];

%Note with will be quiet since need to recalib after HRTF is applied if
%don't use deviceWriter implementation

while ~isDone(fileReader)
    
    
    
    audioIn = fileReader();
    samplesRead = samplesRead + fileReader.SamplesPerFrame;
    
    leftChannel = leftFilter(audioIn,leftIR(sourcePositionIndex,:)) ;
    rightChannel = rightFilter(audioIn,rightIR(sourcePositionIndex,:));
    
    az(sourcePositionIndex)
    
    deviceWriter([leftChannel,rightChannel]*5);
     
    %leftChannel = [leftChannel; leftFilter(audioIn,leftIR(sourcePositionIndex,:)) ];
    %rightChannel = [rightChannel; rightFilter(audioIn,rightIR(sourcePositionIndex,:))];

    if mod(samplesRead,samplesPerPosition) == 0
        sourcePositionIndex = sourcePositionIndex + 1;
    end
end

release(fileReader)
release(deviceWriter)

%Filter assumes same sized input after being set, so have to release and
%remake filter for snapshots.  Shouldn't have to do again though since
%snapshots are all the same size

release(leftFilter)
release(rightFilter)
%% Second Play!

leftFilter = dsp.FIRFilter('NumeratorSource','Input port');
rightFilter = dsp.FIRFilter('NumeratorSource','Input port');

fileReader = dsp.AudioFileReader([ThisTrialFolder '\' filename '.wav']);
deviceWriter = audioDeviceWriter('SampleRate',fileReader.SampleRate);


samplesPerPosition = durationPerPosition*fileReader.SampleRate;
samplesPerPosition = samplesPerPosition - rem(samplesPerPosition,fileReader.SamplesPerFrame);

sourcePositionIndex = 1;
samplesRead = 0;

leftChannel = []; %should preallocate fully for speed but for now this is fine
rightChannel = [];

%Note with will be quiet since need to recalib after HRTF is applied if
%don't use deviceWriter implementation

while ~isDone(fileReader)
    
    
    
    audioIn = fileReader();
    samplesRead = samplesRead + fileReader.SamplesPerFrame;
    
    leftChannel = leftFilter(audioIn,leftIR(sourcePositionIndex,:)) ;
    rightChannel = rightFilter(audioIn,rightIR(sourcePositionIndex,:));
    
    az(sourcePositionIndex)
    
    deviceWriter([leftChannel,rightChannel]*5);
     
    %leftChannel = [leftChannel; leftFilter(audioIn,leftIR(sourcePositionIndex,:)) ];
    %rightChannel = [rightChannel; rightFilter(audioIn,rightIR(sourcePositionIndex,:))];

    if mod(samplesRead,samplesPerPosition) == 0
        sourcePositionIndex = sourcePositionIndex + 1;
    end
end

release(fileReader)
release(deviceWriter)

%Filter assumes same sized input after being set, so have to release and
%remake filter for snapshots.  Shouldn't have to do again though since
%snapshots are all the same size

release(leftFilter)
release(rightFilter)

%% THIRD PLAY!
leftFilter = dsp.FIRFilter('NumeratorSource','Input port');
rightFilter = dsp.FIRFilter('NumeratorSource','Input port');

fileReader = dsp.AudioFileReader([ThisTrialFolder '\' filename '.wav']);
deviceWriter = audioDeviceWriter('SampleRate',fileReader.SampleRate);


samplesPerPosition = durationPerPosition*fileReader.SampleRate;
samplesPerPosition = samplesPerPosition - rem(samplesPerPosition,fileReader.SamplesPerFrame);

sourcePositionIndex = 1;
samplesRead = 0;

leftChannel = []; %should preallocate fully for speed but for now this is fine
rightChannel = [];

%Note with will be quiet since need to recalib after HRTF is applied if
%don't use deviceWriter implementation

while ~isDone(fileReader)
    
    
    
    audioIn = fileReader();
    samplesRead = samplesRead + fileReader.SamplesPerFrame;
    
    leftChannel = leftFilter(audioIn,leftIR(sourcePositionIndex,:)) ;
    rightChannel = rightFilter(audioIn,rightIR(sourcePositionIndex,:));
    
    az(sourcePositionIndex)
    
    deviceWriter([leftChannel,rightChannel]*5);
     
    %leftChannel = [leftChannel; leftFilter(audioIn,leftIR(sourcePositionIndex,:)) ];
    %rightChannel = [rightChannel; rightFilter(audioIn,rightIR(sourcePositionIndex,:))];

    if mod(samplesRead,samplesPerPosition) == 0
        sourcePositionIndex = sourcePositionIndex + 1;
    end
end

release(fileReader)
release(deviceWriter)

%Filter assumes same sized input after being set, so have to release and
%remake filter for snapshots.  Shouldn't have to do again though since
%snapshots are all the same size

release(leftFilter)
release(rightFilter)

clear fileReader
clear deviceWriter
clear leftFilter
clear rightFilter

%% Reset filters for use on snapshots
%Filter assumes same sized input after being set, so have to release and
%remake filter for snapshots.  Shouldn't have to do again though since
%snapshots are all the same size


leftFilter = dsp.FIRFilter('NumeratorSource','Input port');
rightFilter = dsp.FIRFilter('NumeratorSource','Input port');

clf

%%  Give Instructions and Run Actual Tasks

figure(1)
text(0.45,0.8,'We will now present test stimuli.','Color','red','FontSize',14);
axis off
pause(4)
figure(1)
text(0.45,0.7,'Please indicate whether this test stimulus was part of or not part of the reference stimulus.','Color','red','FontSize',14);
axis off
% figure(1)
% text(.1,0.6,'or is not a part of the loop you just heard.','Color','red','FontSize',14);
% axis off

pause(4)

clf
figure(1)
text(0.45,0.8,'Use the gamepad to indicate your choice.','Color','red','FontSize',14);
axis off
pause(4)
figure(1)
text(0.45,0.7,'Hit the left shoulder button to indicate the clip is not a part of the loop you just heard.','Color','red','FontSize',14);
axis off
pause(4)
figure(1)
text(0.45,0.6,'Try hitting the left shoulder button now.','Color','red','FontSize',14);
axis off

while button(joy,7)<1
    
    pause(0.05)
    
end

figure(1)
text(0.45,0.5,'Good!','Color','red','FontSize',14);
axis off
pause(2)
figure(1)
text(0.45,0.4,'Hit the right shoulder button to indicate the clip was a part of the loop you just heard.','Color','red','FontSize',14);
axis off
pause(2)
figure(1)
text(0.45,0.3,'Try hitting the right shoulder button now.','Color','red','FontSize',14);
axis off

while button(joy,8)<1
    
    pause(0.05)
    
end

figure(1)
text(0.45,0.2,'Good!','Color','red','FontSize',14);
axis off
pause(1)
clf

figure(1)
text(0.45,0.8,'Remember:','Color','red','FontSize',14);
axis off
figure(1)
text(0.45,0.7,'Hit the left button if you think the test stimulus is not part of the reference stimulus','Color','red','FontSize',14);
axis off
figure(1)
text(0.45,0.6,'Hit the right button if you think it is part of the reference stimulus','Color','red','FontSize',14);
axis off
pause(4)

clf
figure(1)
text(0.45,0.8,'The task will now begin','Color','red','FontSize',14);
axis off
pause(2)
figure(1)
text(0.45,0.7,'A test stimulus will play and then you will have 5 seconds to respond','Color','red','FontSize',14);
axis off
figure(1)
text(0.45,0.6,'Hit any button on the gamepad to initiate each trial','Color','red','FontSize',14);
axis off

while all((button(joy))<1)
    
    pause(0.05)
    
end

clf
%%
Answer_Key = -1 * ones(trials,1);
Responses = NaN*ones(trials,1);

t = timer('TimerFcn', 'status=false; disp(''...'')',... 
                 'StartDelay',5);
figure(1)

for i = 1:20 %trials
    
    i
    
    ind = NewOrder(i);
    
    Answer_Key(i) = save_snaps{ind}.altered;
    
    clf
    text(0.45,0.7,['Trial ' num2str(i)],'Color','red','FontSize',14);
    text(0.45,0.4, 'Respond when the test stimulus finishes','Color','red','FontSize',14)
    text(0,0.4,'<- Not from reference','Color','red','FontSize',14);
    text(0.8,0.4,' From reference ->','Color','red','FontSize',14)
    axis off
    
    
    pause(1)
    
 
%*************************************************************************%
%this needs to be changed to play a snapshot from a different location each
%time:

%2/25/19 two options, either save all 100 as separate wav files or try to do HRTF
%online with save_snaps.snap_piece

%2/26/19 went with former option since it was faster to implement

    %save_snaps{NewOrder(i)}.audio.play
    
fileReader = dsp.AudioFileReader([ThisTrialFolder '\snap_wavs\snap_' num2str(NewOrder(i)) '.wav']);
deviceWriter = audioDeviceWriter('SampleRate',fileReader.SampleRate);
    
samplesPerPosition = durationPerPosition*fileReader.SampleRate;
samplesPerPosition = samplesPerPosition - rem(samplesPerPosition,fileReader.SamplesPerFrame);

sourcePositionIndex = randsample(length(az),1);
samplesRead = 0;

while ~isDone(fileReader)
    audioIn = fileReader();
    samplesRead = samplesRead + fileReader.SamplesPerFrame;

    leftChannel = leftFilter(audioIn,leftIR(sourcePositionIndex,:)) ;
    rightChannel = rightFilter(audioIn,rightIR(sourcePositionIndex,:));

    %az(sourcePositionIndex)
    
    deviceWriter([leftChannel,rightChannel]*5);

    if mod(samplesRead,samplesPerPosition) == 0
        sourcePositionIndex = sourcePositionIndex + 1;
    end
end

release(deviceWriter)
release(fileReader)
    
    
    
    
   
   
%*************************************************************************%    
    status = true;
    
    start(t)
    
    while status == true && all(button(joy,[7,8])<1)
        
    thisresponse = button(joy,[7,8]);   
        
    end
    %to try to catch it if you press too early by setting this to the
    %button you are holding
    
    if any(button(joy,[7,8])>0)
        
         thisresponse = button(joy,[7,8]); 
        
    end
    
    %if one of the shoulder buttons was pressed record whether or not left
    %was pressed (since left indicates altered)
    %if nothing is pressed leave nan value for that trial
    if any(thisresponse==1)
    
    Responses(i) = thisresponse(1);
    
    end
    
    clf
    
    if Responses(i) == Answer_Key(i)
        
    figure(1)
    text(0.45,0.6,'Correct','Color','green','FontSize',14);
    axis off
    
    elseif isnan(Responses(i))
        
    figure(1)
    text(0.45,0.6,'No Response Recorded','Color','red','FontSize',14);
    axis off   
        
    else
       
    figure(1)
    text(0.45,0.6,'Incorrect','Color','red','FontSize',14);
    axis off
        
    end
    
    clear thisresponse
    pause(4)
    
    clf
    figure(1)
    text(0.45,0.6,'Hit any button on the gamepad to initiate the next trial','Color','red','FontSize',14);
    axis off
    
    while all((button(joy))<1)
    
    pause(0.05)
    
    end
    
end

clf
figure(1)
text(0.35,0.45,'Thanks for Participating!','Color','Green','FontSize',20);
axis off


100*sum(Responses == Answer_Key)/length(Answer_Key)



%% Save outputs %note: add something to this to change file name given above arguments

filename = [ strcat(date,'_Loop1_C_') 'Responses_and_AnswerKey'];
save([ThisTrialFolder '\' filename], 'Responses', 'Answer_Key')



%% Testing out snapshot with HRTF stuff

% indss = 1;
% 
% %Filter assumes same sized input after being set, so have to release and
% %remake filter for snapshots.  Shouldn't have to do again though since
% %snapshots are all the same size
% 
% %release(leftFilter)
% %release(rightFilter)
% 
% %leftFilter = dsp.FIRFilter('NumeratorSource','Input port');
% %rightFilter = dsp.FIRFilter('NumeratorSource','Input port');
% 
% fileReader = dsp.AudioFileReader([ThisTrialFolder '\snap_wavs\snap_' num2str(NewOrder(snap_ind)) '.wav']);
% deviceWriter = audioDeviceWriter('SampleRate',fileReader.SampleRate);
% 
% 
% samplesPerPosition = durationPerPosition*fileReader.SampleRate;
% samplesPerPosition = samplesPerPosition - rem(samplesPerPosition,fileReader.SamplesPerFrame);
% 
% sourcePositionIndex = randsample(length(az),1);
% samplesRead = 0;
% 
% while ~isDone(fileReader)
%     audioIn = fileReader();
%     samplesRead = samplesRead + fileReader.SamplesPerFrame;
% 
%     leftChannel = leftFilter(audioIn,leftIR(sourcePositionIndex,:)) ;
%     rightChannel = rightFilter(audioIn,rightIR(sourcePositionIndex,:));
% 
%     az(sourcePositionIndex)
%     
%     deviceWriter([leftChannel,rightChannel]*5);
% 
%     if mod(samplesRead,samplesPerPosition) == 0
%         sourcePositionIndex = sourcePositionIndex + 1;
%     end
% end
% 
% release(deviceWriter)
% release(fileReader)
% 

%% Testing out applying filters off line then playing sound
%Note this is actually quite slow, may be reading in file in real time so
%would take 5 minutes to do...seems like it is reading the file in real
%time which is lame, but just have to deal with this for now

%This is kind of unacceptable, but can be done ahead of time to prevent
%issues for participants, for now it is fine if it works

%Since only need to copy first l-r pass, could simply run it that many
%times then loop those chunks with volume ramp applied for clicking

%Clicks start around 120 seconds for no reason even if apply filter
%offline...thinking may just apply filter to first minute and then copy
%that to get to five minutes

%Set up now to make a loop that is 100 seconds long and will simply repeat
%below 3 times to get to five minutes as before

% leftFilter = dsp.FIRFilter('NumeratorSource','Input port');
% rightFilter = dsp.FIRFilter('NumeratorSource','Input port');
% 
% fileReader = dsp.AudioFileReader([ThisTrialFolder '\' filename '.wav']);
% deviceWriter = audioDeviceWriter('SampleRate',fileReader.SampleRate);
% 
% 
% samplesPerPosition = durationPerPosition*fileReader.SampleRate;
% samplesPerPosition = samplesPerPosition - rem(samplesPerPosition,fileReader.SamplesPerFrame);
% 
% sourcePositionIndex = 1;
% samplesRead = 0;
% 
% leftChannel = []; %should preallocate fully for speed but for now this is fine
% rightChannel = [];
% 
% %Note with will be quiet since need to recalib after HRTF is applied if
% %don't use deviceWriter implementation
% 
% while ~isDone(fileReader)
%     
%     
%     
%     audioIn = fileReader();
%     samplesRead = samplesRead + fileReader.SamplesPerFrame;
%     
%     leftChannel = leftFilter(audioIn,leftIR(sourcePositionIndex,:)) ;
%     rightChannel = rightFilter(audioIn,rightIR(sourcePositionIndex,:));
%     
%     az(sourcePositionIndex)
%     
%     deviceWriter([leftChannel,rightChannel]*5);
%      
%     %leftChannel = [leftChannel; leftFilter(audioIn,leftIR(sourcePositionIndex,:)) ];
%     %rightChannel = [rightChannel; rightFilter(audioIn,rightIR(sourcePositionIndex,:))];
% 
%     if mod(samplesRead,samplesPerPosition) == 0
%         sourcePositionIndex = sourcePositionIndex + 1;
%     end
% end
% 
% release(fileReader)
% release(deviceWriter)
% 
% %Filter assumes same sized input after being set, so have to release and
% %remake filter for snapshots.  Shouldn't have to do again though since
% %snapshots are all the same size
% 
% release(leftFilter)
% release(rightFilter)
