%% 2019-06-03

%First pass just trying to get manipulating chirp function down so that it
%works this way

%% Copied from old code: Make a simply trajectory

PM =1; CM = 0; %just to set up doing parallel motion for now

%Base Parameters
Fs = 44100; %Pulled from Adam's code
dt = 1/Fs; %Resolution from sampling frequency
Full_Duration = 5; %seconds
Midpoint = floor(Full_Duration/2); %middle of signal in real (nonnegative) time, "zeros" of quadratic
T = -(Full_Duration/2):dt:Full_Duration/2; %vector of time stamps for full duration of stimulus
%Note will have to change this if want different sweep structure, but
%leaving above to use quadratic chirp function to prevent discontinuity


%Stimulus Parameters
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


%(will probably make switch-case later and or make separate functions)
%amplitude_motion=0; %Have transitions over amplitude
Frequency_motion=1; %Have transitions over frequency (this is what is in research aims)

parallel_motion=PM; %Have all frequencies move together
contrary_motion=CM; %Have frequencies move oppositely
random_motion = 0; %Have random base frequencies selected to move randomly

Stim.Fs = Fs;
Stim.Duration = Full_Duration;
Stim.SF = [SF1 SF2 SF3];


if Frequency_motion
    amp = 0.05; %keeps it from blowing out the headphones
    max_change = 1.5; %Frequency at peak or trough of quadratic
    
    if parallel_motion
        
        %For now just doing simple quadratic motion of frequency
        
        %to remove discontinuity, made frequency changes quadratic
        %NOTE IF CHANGE LENGTH OF FULL DURATION HAVE to change reference
        %time
        %^fixed above
        
        Piece1 = amp*chirp(T,SF1*max_change,Midpoint,SF1,'quadratic'); 
        Piece2 = amp*chirp(T,SF2*max_change,Midpoint,SF2,'quadratic');
        Piece3 = amp*chirp(T,SF3*max_change,Midpoint,SF3,'quadratic');
        
        Full_stim = Piece1+Piece2+Piece3;
        Stim.Full_stim = Full_stim;
        Stim.Piece1 = Piece1;
        Stim.Piece2 = Piece2;
        Stim.Piece3 = Piece3;
        
    end
    
    
end

%% Now same trajectory at different rates

%Most basic way: simply decrease T and Midpoint, issue now have stimulus
%for different duration
%Note: named fast just because speeding up seems like logic first thing to
%do

percent_dur =.25; %percent of Full_Duration to use (i.e. conversely, the inverse of how much faster it is (e.g. half of the time is double the speed)
Full_Duration_fast = Full_Duration * percent_dur;
Midpoint_fast = Full_Duration_fast/2;
T_fast = -(Full_Duration_fast/2):dt:Full_Duration_fast/2;

    Piece1_fast = amp*chirp(T_fast,SF1*max_change,Midpoint_fast,SF1,'quadratic'); 
    Piece2_fast = amp*chirp(T_fast,SF2*max_change,Midpoint_fast,SF2,'quadratic');
    Piece3_fast = amp*chirp(T_fast,SF3*max_change,Midpoint_fast,SF3,'quadratic');

    Full_stim_fast = Piece1_fast+Piece2_fast+Piece3_fast;
    Stim_fast.Full_stim = Full_stim;
    Stim_fast.Piece1 = Piece1_fast;
    Stim_fast.Piece2 = Piece2_fast;
    Stim_fast.Piece3 = Piece3_fast;

%% Basic listening 1

%2019-06-03 basic trick works, struggles if Midpoint_fast rounds to zero,
%removing rounding to see effect

audio_og = audioplayer(Full_stim,Fs, 24);

audio_different = audioplayer(Full_stim_fast,Fs, 24);

audio_og.play

pause(7)

audio_different.play

%% quick spectrogram

figure(1)
spectrogram(Full_stim,Fs/10,Fs/20,Fs/10,Fs, 'yaxis')
axis([0 Full_Duration 0 6 ])
title('Standard Speed')


figure(2)
spectrogram(Full_stim_fast,Fs/10,Fs/20,Fs/10,Fs, 'yaxis')
axis([0 Full_Duration_fast 0 6 ])
title('Altered Speed')

%% Other parabolas: i.e. just seeing how it sounds when we have same SF but move the inflection point

max_change_new = 2.25;

Piece1_new = amp*chirp(T,SF1*max_change_new,Midpoint,SF1,'quadratic'); 
Piece2_new = amp*chirp(T,SF2*max_change_new,Midpoint,SF2,'quadratic');
Piece3_new = amp*chirp(T,SF3*max_change_new,Midpoint,SF3,'quadratic');

Full_stim_new = Piece1_new+Piece2_new+Piece3_new;
Stim_new.Full_stim = Full_stim_new;
Stim_new.Piece1 = Piece1_new;
Stim_new.Piece2 = Piece2_new;
Stim_new.Piece3 = Piece3_new;

%% Basic listening 2

%2019-06-03 basic trick works, struggles if Midpoint_fast rounds to zero,
%removing rounding to see effect

audio_og = audioplayer(Full_stim,Fs, 24);

audio_different = audioplayer(Full_stim_new,Fs, 24);

audio_og.play

pause(7)

audio_different.play

