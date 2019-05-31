function [ Stim ] = GenStimulus(PM,CM)
%GENSTIMULUS Function to generate base stimulus 
%   Detailed explanation goes here

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
    
    if contrary_motion == 1 
        
        
        
        Piece1 = amp*chirp(T,SF1*max_change,Midpoint,SF1,'quadratic'); 
        Piece2 = amp*chirp(T,SF2,Midpoint,SF2*max_change,'quadratic');
        Piece3 = amp*chirp(T,SF3*max_change,Midpoint,SF3,'quadratic');
       
        Full_stim = Piece1+Piece2+Piece3;
       
        Stim.Full_stim = Full_stim;
        Stim.Piece1 = Piece1;
        Stim.Piece2 = Piece2;
        Stim.Piece3 = Piece3;
        
    end
    
    if random_motion == 1
        
        %set this up later
        
    end
    
    Stim.MC = max_change;
    Stim.amp = 0.05;
    
    
end

