%% 2019-06-14
%Cleaned up and consolidated code to put on github/pass to Chetan

%input: none currently, usually write this as a function but will leave as
%a script for development

%output of first block: struct, "Stim" with fields:
%Fs - sampling frequency
%Full_Duration - time in seconds for single presentation of stimulus
%SF - 3 starting frequencies of parabola sweep
%Full_stim - the actual stimulus
%Piece1, Piece2, Piece3 - the three composite pieces of the full stimulus

%later blocks modify (or have the starting code for modifying) basic stimulus

%end of each block has some code to play the stimulus

%last block has some code to turn into wav file

%note to self will have to recalibrate headphones/think carefully about
%normalization for amplitude modulated signals if want peak loudness to be
%the same across stimuli

%% Generate Basic Stimulus

%Basic Parameters
Fs = 44100; %Sampling Rate, Pulled from Adam's code
dt = 1/Fs; %Resolution from sampling frequency
Full_Duration = 5; %seconds
Midpoint = floor(Full_Duration/2); %middle of signal in real (nonnegative) time, "zeros" of parabola
T = -(Full_Duration/2):dt:Full_Duration/2; %vector of time stamps for full duration of stimulus

%Frequencies for stimulus
Frequency_bank = [285 350 397 440 499 569 650 788 845 875 949 1000 1100 ]; %In NHP experiment will be based on the BF for the neural population,
                     %for now just setting to a decent range of random
                     %selected frequencies.  Later can do something more
                     %elborate.
                     
SF1 = Frequency_bank(randi(length(Frequency_bank),1)); %(Hz) Frequency of tone the stimulus trajectory starts at
SF2 = Frequency_bank(randi(length(Frequency_bank),1)); %(Hz) Frequency of tone the stimulus trajectory starts at
SF3 = Frequency_bank(randi(length(Frequency_bank),1)); %(Hz) Frequency of tone the stimulus trajectory starts at

%Basic checks for selected frequencies:

%make sure note whole number ratios of one another (or equal)
Freq_Mat = meshgrid([SF1, SF2, SF3])./meshgrid([SF1, SF2, SF3])';
Freq_Mat = Freq_Mat - diag(diag(Freq_Mat))*.1; %removes trivial case of diagonals being one

while any(any(floor(Freq_Mat)==Freq_Mat))

SF1 = Frequency_bank(randi(length(Frequency_bank),1)); %(Hz) Frequency of tone the stimulus trajectory starts at
SF2 = Frequency_bank(randi(length(Frequency_bank),1)); %(Hz) Frequency of tone the stimulus trajectory starts at
SF3 = Frequency_bank(randi(length(Frequency_bank),1)); %(Hz) Frequency of tone the stimulus trajectory starts at

Freq_Mat = meshgrid([SF1, SF2, SF3])./meshgrid([SF1, SF2, SF3])';
Freq_Mat = Freq_Mat - diag(diag(Freq_Mat))*.1; %removes trivial case of diagonals being one

end

%Create parabolas and combine into full stimulus

amp = 0.05; %rescales amplitude to keep it from blowing out the headphones
max_change = 1.5; % max_change * Starting Frequencies is peak of parabola
%by consequence, this sets the focus of the parabola

%chirp function acts in frequency-time space (i.e. spectrogram) 
%T sets time domain, second argument is focus, third is amount of time to
%focus, forth is the zeros

Piece1 = amp*chirp(T,SF1*max_change,Midpoint,SF1,'quadratic'); 
Piece2 = amp*chirp(T,SF2*max_change,Midpoint,SF2,'quadratic');
Piece3 = amp*chirp(T,SF3*max_change,Midpoint,SF3,'quadratic');
Full_stim = Piece1+Piece2+Piece3; %simply combine signals

%Collect important outputs into Stim which is cell of structs

Stim = cell(1,1);

Stim{1}.Full_stim = Full_stim;
Stim{1}.Piece1 = Piece1;
Stim{1}.Piece2 = Piece2;
Stim{1}.Piece3 = Piece3;
Stim{1}.Fs = Fs;
Stim{1}.Duration = Full_Duration;
Stim{1}.SF = [SF1 SF2 SF3];

%% make sci-fi comp wav file

% ending = length(Full_stim);
% starting = 1;
% 
% binsize = .200*Fs;
% bin_num = round(length(Full_stim)/binsize);
% 
% bininds = 1:binsize:length(Full_stim);
% 
% 
% 
% rand_grab = randperm(bin_num,bin_num);
% 
% shuffle_stim = zeros(1,length(Full_stim));
% 
% 
% 
% 
% for i=1:length(bininds)-1
% 
% shuffle_stim(bininds(i):bininds(i+1)) = Full_stim(bininds(rand_grab(i)):bininds(rand_grab(i)+1));
% 
% end

%% Play stimulus

Mod0_normal=audioplayer(Full_stim,Fs,24);

Mod0_normal.play
% 
% pause(7)

% Modfunny = audioplayer(shuffle_stim,Fs,24);
% Modfunny.play


%% Generate Wav files
%have to do this manually for now, will set up something clever later
%in short, have to save each parabola (i.e. cell of Stim cell array
%as its own separate wav

% %change name and such as you see fit for each one
% 
% name = 'importantnoises3';
% 
% 
% %For psychophysics we would have each one be presented for 5 minutes
% %Loop_Duration = 5; %in minutes
% %Loop = repmat(Stim{index}.Full_stim,1,ceil(60*Loop_Duration)/Full_Duration);
% 
% audiowrite([name '.wav'], shuffle_stim, Fs,'BitsPerSample',24) %first save the loop

%% Modification 1: Change Foci

%Pretty straight foward, just change max_change parameter from earlier code
%block

max_change_distrib = 1.75:.25:3; %Distribution decided from just listening to different max_changes

n_parabolas = 3; %number of new parabolas to make with new foci

max_change_samples = max_change_distrib(randsample(length(max_change_distrib),n_parabolas));
%Pull max_change from distrib

%Make new parabolas

for i = 1:n_parabolas
 
    %no one cares about preallocating with these few elements
    amp = 0.05; %rescales amplitude to keep it from blowing out the headphones
    max_change = 1.5; % max_change * Starting Frequencies is peak of parabola
%by consequence, this sets the focus of the parabola

%chirp function acts in frequency-time space (i.e. spectrogram) 
%T sets time domain, second argument is focus, third is amount of time to
%focus, forth is the zeros

    Piece1 = amp*chirp(T,SF1*max_change_samples(i),Midpoint,SF1,'quadratic'); 
    Piece2 = amp*chirp(T,SF2*max_change_samples(i),Midpoint,SF2,'quadratic');
    Piece3 = amp*chirp(T,SF3*max_change_samples(i),Midpoint,SF3,'quadratic');
    Full_stim = Piece1+Piece2+Piece3; %simply combine signals
    
    %add to Stim by appending cell that contains new struct
    %Calling length of stim so Modification blocks can be ran in any order
    inds = length(Stim)+1;
    Stim{inds}.Full_stim = Full_stim;
    Stim{inds}.Piece1 = Piece1;
    Stim{inds}.Piece2 = Piece2;
    Stim{inds}.Piece3 = Piece3;
    Stim{inds}.Fs = Fs;
    Stim{inds}.Duration = Full_Duration;
    Stim{inds}.SF = [SF1 SF2 SF3];
    Stim{inds}.max_change = max_change_samples(i);    
end

%% Play Example vs OG

Mod0_normal.play

Mod1_Foci = audioplayer(Stim{2}.Full_stim,Fs,24); %only pulling last one as example

pause(Full_Duration+2) %wait so they don't play one after another

Mod1_Foci.play


%% Modification 2: Amplitude Modulation
%Have modulating amplitude (think tremolo) of each parabola
%Probably will be easiest just do do this piece wise since pretty
%sure need to have single frequencey carrier for this equation to work

%Also need to rewrite later to do this multiple times instead of just once,
%but for now just leave as is

%Message signal parameters
alpha = .5; %modulation index, unitless, (0 to 1) where 0 is no modulation
Mess_freq_distribution = 1:.5:10; %apparently <20pi or <10 Hz is tremolo effect range (check this)
%update: checked at perceptually seems true, at least 20 or more is
%definitely not just tremolo any more, and 15 is getting there

Fm = 2; Mess_freq_distribution(randsample(length(Mess_freq_distribution),1)); %pull a random frequency 

phim = 0; %phase shift phi, units of these are seconds for now, keeping zero for now
t=0:dt:Full_Duration; %Message signal should be as long as stimulus (NOTE MAY HAVE TO CHANGE THIS LATER WHEN CHANGE SPEED)

%Create message signal, right now sine wave just because it is easy, but
%will add subtler effects soon (e.g. have a ramp for half or something)

Message = sin(2*pi*Fm*(t+phim)); %note message comes from idea behind AM radio, tune to carrier but modulations on carrier hold the actual information

%Add to each carrier (i.e each piece) of Full_stim;

%Also assume just doing this on first parabola, change index to change this
%Using basic AM mod equation (relies on interference) og + message*og
%og is usually called carrier

index = 1;

Piece1 = Stim{index}.Piece1 + alpha.*Message.*Stim{index}.Piece1;
Piece2 = Stim{index}.Piece2 + alpha.*Message.*Stim{index}.Piece2;
Piece3 = Stim{index}.Piece3 + alpha.*Message.*Stim{index}.Piece3;

Full_stim = Piece1 + Piece2 + Piece3; %recombine

%Same as before, set so add in new cells to Stim which contain structs
inds = length(Stim)+1;
%Fill in necessary details
 Stim{inds}.Full_stim = Full_stim;
 Stim{inds}.Piece1 = Piece1;
 Stim{inds}.Piece2 = Piece2;
 Stim{inds}.Piece3 = Piece3;
 Stim{inds}.Fs = Fs;
 Stim{inds}.Duration = Full_Duration;
 Stim{inds}.SF = [SF1 SF2 SF3];
 Stim{inds}.AM_freq = Fm;
 Stim{inds}.AM_index = alpha;
 Stim{inds}.AM_phi = phim;
 Stim{inds}.AM_shape = 'Sine Wave'; %change as needed
 
%% Play Example vs OG

Mod0_normal.play

Mod2_AM = audioplayer(Stim{inds}.Full_stim,Fs,24); %only pulling last one as an example

pause(Full_Duration+2) %wait so they don't play one after another

Mod2_AM.play




%% Modification 3: "Speed" (rate of frequency change along parabola) Modulation (under development)

%NOT ADDED YET SINCE STILL NEED TO LOOK THROUGH GEFFEN PAPER THIS WEEKEND
%There is some code in Matlab 2019A but not in what we have

%Update: 2019-06-19
%Found some good starter code from Bill Sethares (sp) for a phase vocoder
%(see the pdf of his book we have or his website to get math overview behind
%this method)



%% Modification 4: Try making comodulated noise

%Step one AM modulated noise

%Just doing a quick pull from Nelken stuff to try to pin down at least
%semi-realistic ranges
%update: this works to generate at least the base noise
%update: put in comod, hear a difference unclear if it is desired
%difference
%DOuble check that this works given that the same thing being applied to
%the tone sweeps did not work as expected...

n_elements = length(Full_stim); %to generate white noise need to just draw samples, sample number will be determined by length of stimulus


%just leaving it as white noise with variance .01)
bp_w_noise = randn([n_elements,1])/10;

bp_w_noise = bandpass(bp_w_noise, [220 8000], Fs); %pulled from Nelken paper but that is with crows so slightly suspicious but has noise that can be from ~2500 to 5000hz
%note this line seems to take a sec to process...


%test it out
% audio_noise = audioplayer(bp_w_noise,Fs,24);

%now do the comodulations with 7 sines of differeing phases and
%frequencies.  Taking this Amp mod as same as temp mod and borrowing
%frequency distribution from Singh and Theunissen (not using this distrib
%directly though, just setting things uniform for now, especiallyh since
%exact distrib is unclear) uniform discrete with only integer value just
%for a  bit now, but no real issues.  Add in bit to ensure dont get same
%frequencies
%later.****************************************************************

%for now have 6 random sines added with random phases
freq = randsample(10,6); %env sounds look like that may mostly cap out at 200 Hz 
%random phase shift is just a random percent of period

%If only want tremolo effects < 10hz
%freq = randsample(10,6);

%If want just strong tremolo, this and no phase shift
% freq = 5;
% AM = sin(2*pi*freq(1)*0:n_elements-1);

alpha = 1; %modulation index, unitless, (0 to 1) where 0 is no modulation

AM = [sin(2*pi*freq(1) * 0:n_elements-1) ... % + (freq(1)^-1 * rand(1)));...
    sin(2*pi*freq(2)* 0:n_elements-1) ... %+ (freq(2)^-1 * rand(1)));...
    sin(2*pi*freq(3)* 0:n_elements-1) ...  + (freq(3)^-1 * rand(1)));...
    sin(2*pi*freq(4)* 0:n_elements-1) ... + (freq(4)^-1 * rand(1)));...
    sin(2*pi*freq(5)* 0:n_elements-1) ... + (freq(5)^-1 * rand(1)));...
    sin(2*pi*freq(6)* 0:n_elements-1)]; ... + (freq(6)^-1 * rand(1)))];
AM = sum(AM)';
%%For simple test case
%AM = sin(2*pi*freq(1)*0:n_elements-1)'; %have to transpose otherwise run out of memory

%rescale so bp_w_noise is on sameish scale (just determined by peak of amp
%signal)

index = 1;

comod_bp_w_noise = bp_w_noise + alpha.*AM .* bp_w_noise; %same formula as above for AM signal
comod_bp_w_noise = comod_bp_w_noise/max(comod_bp_w_noise) * max(Stim{index}.Full_stim);

%moved this to the end since I am worried scaling is causing issues.
bp_w_noise = (bp_w_noise/max(bp_w_noise) * max(Full_stim))'; 

%% Test

audio_noise_plain = audioplayer(bp_w_noise,Fs,24);

audio_noise_plain.play

pause(7)

audio_noise_cmod = audioplayer(comod_bp_w_noise,Fs,24);

audio_noise_cmod.play


%divide each amplitdue by the sqrt of the bandwidth
%Fix this
%Update 2019-06-27 mostly fixed, perceptual effect seems to be slight but
%could be there.  Thinking on how to visulized to see that it is there
%% Modification 5: Reverb

%Add this using audacity after have generated wav files and then just
%resave

%Traer and McDermott 2016 makes for some interesting reading on the
%statistical properties of natural reverbration,
%but this is persumably handled by audacity in its algorithm.
%Still interesting to mull on though, particularly in the context of SFA

%% Generate Wav files
%have to do this manually for now, will set up something clever later
%in short, have to save each parabola (i.e. cell of Stim cell array
%as its own separate wav

%change name and such as you see fit for each one

name = 'test1';
index = 1;

%For psychophysics we would have each one be presented for 5 minutes
%Loop_Duration = 5; %in minutes
%Loop = repmat(Stim{index}.Full_stim,1,ceil(60*Loop_Duration)/Full_Duration);

audiowrite([name '.wav'], Stim{index}.Full_stim, Stim{1}.Fs,'BitsPerSample',24) %first save the loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%^^^Change to Loop if doing loops%%%***********