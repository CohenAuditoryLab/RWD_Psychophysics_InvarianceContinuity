%% This file runs the counters to the full task via MATLAB
%May be updated to run on Tower of Babel next but for now does the
%following:

%0) Assumes calibration has been done and calibration files are on path
%(acutally in folder is best for now)

%1) Loads previously generated reference stimulus and snapshots from the
%other stimulus conditions. I.e. will load continuous stimulus but test on
%non-continuous snap shots or will load non-continuous snap shots and test on
%continuous snap shots

%2) Runs the actual task with figures appearing as prompts and records
%answers

%3) Save answers with file

%% Set Conditions: either continuous reference with discontinous snap shots or vice versa

Con_with_Discon = 0; 

Discon_with_Con = 1;

trials = 100; %just hard coding for now for convenience

%% Continuous reference stimuli testing with discontinuous snap shots

if 1 == Con_with_Discon
%% set base stimulus file to load in***change each time

ThisTrialFolder = [pwd '\Trials\Trial_1'];

date_file = '15-Mar-2019'; %just change this***************************

filename = [date_file '_Loop1_C'];

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
   
    %% Ready Joystick

%Note for all below joystick = gamepad controller

% Define joystick ID
ID = 1;
% Create joystick variable
joy=vrjoystick(ID);

    %% Listen to Loop
    
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

%% Play stimulus

%As noted elsewhere, repeat this process three times to play clean clip
%since only get 100s of clean playing, (300s = 5 minutes)
% FIRST PLAY!
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

% THIRD PLAY!
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


    %%  Give Instructions

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
%% Load in and randomly change order of snapshots

ThisTrialFolder = [pwd '\Trials\Trial_2'];

date_file = '22-Mar-2019';

load([ThisTrialFolder '\' date_file '_Loop1_DC_snapshot_variables.mat']) %pull snap shot so can make answer key

NewOrder = randsample(100,100); %Pull a new order again just to be safe


Answer_Key = -1 * ones(trials,1);
Responses = NaN*ones(trials,1);

t = timer('TimerFcn', 'status=false; disp(''...'')',... 
                 'StartDelay',5);
figure(1)

for i = 1:100 %trials
    
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

    %% Save outputs********************rewrite each time
    
    %also for now have to manually make dir for new folder
    
    ThisTrialFolder = [pwd '\Trials\Trial_3'];
    
    filename = [ strcat(date,'_Loop1_C_DC_Mismatch_') 'Responses_and_AnswerKey'];
    save([ThisTrialFolder '\' filename], 'Responses', 'Answer_Key', 'NewOrder')
    
end
%% ***********************************************************************************************
%%************************************************************************************************
%%************************************************************************************************
%% Code up discon with con next

if 1 == Discon_with_Con
%% set base stimulus file to load in***change each time

ThisTrialFolder = [pwd '\Trials\Trial_2'];

date_file = '22-Mar-2019'; %just change this***************************

filename = [date_file '_Loop1_DC'];

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
   
    %% Ready Joystick

%Note for all below joystick = gamepad controller

% Define joystick ID
ID = 1;
% Create joystick variable
joy=vrjoystick(ID);

    %% Listen to Loop
    
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

%% Play stimulus

%As noted elsewhere, repeat this process three times to play clean clip
%since only get 100s of clean playing, (300s = 5 minutes)
% FIRST PLAY!
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

% THIRD PLAY!
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


    %%  Give Instructions

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
%% Load in and randomly change order of snapshots

ThisTrialFolder = [pwd '\Trials\Trial_1']; %ALSO HAVE TO CHANGE THESE EACH TIME

date_file = '15-Mar-2019';

load([ThisTrialFolder '\' date_file '_Loop1_C_snapshot_variables.mat']) %pull snap shot so can make answer key

NewOrder = randsample(100,100); %Pull a new order again just to be safe


Answer_Key = -1 * ones(trials,1);
Responses = NaN*ones(trials,1);

t = timer('TimerFcn', 'status=false; disp(''...'')',... 
                 'StartDelay',5);
figure(1)

for i = 1:100 %trials
    
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

    %% Save outputs********************rewrite each time
    
    %also for now have to manually make dir for new folder
    
    ThisTrialFolder = [pwd '\Trials\Trial_4'];
    
    filename = [ strcat(date,'_Loop1_DC_C_Mismatch_') 'Responses_and_AnswerKey'];
    save([ThisTrialFolder '\' filename], 'Responses', 'Answer_Key', 'NewOrder')
    
end

