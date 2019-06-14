%% This is to test that we can get basic AM working and then if that works
%move on/fix what is in the developlment code right now

%Update 2019-06-12 this works perfectly well!
%% Stuff
%Basic time stuff
Fs = 44100; %Pulled from Adam's code
dt = 1/Fs; %Resolution from sampling frequency
Full_Duration = 5; %seconds
t= 0:dt:Full_Duration;
%Carrier signal stuff
Ac = 1;
Fc= 440;
phic = 0;

%Message signal stuff
alpha = .5; 
Fm = 7; %apparently <20pi or <10 Hz is tremolo effect range (check this)
%update: checked at perceptually seems true, at least 20 or more is
%definitely not just tremolo any more, and 15 is getting there

phim = 0; %units of these are seconds for now

%% Signals
%equation comes from multiple sources, can think about later why this
%exactly works but kind of makes sense...
%Update, makes perfect sense, just using interference to do your
%modulation!

Carrier = Ac*sin(2*pi*Fc*(t+phic));

Message = sin(2*pi*Fm*(t+phim)); %note message comes from idea behind AM radio, tune to carrier but modulations on carrier hold the actual information

AModded_Sig = Carrier + alpha.*Message.*Carrier;

subplot(3,1,1)
plot(t(1:Fs/10),Carrier(1:Fs/10))

subplot(3,1,2)
plot(t,Message)

subplot(3,1,3)
plot(t(1:Fs/10),AModded_Sig(1:Fs/10))

%% Zoomed in plot
figure(2)
plot(t(1:Fs/10), AModded_Sig(1:Fs/10))

%% Play sound

Test1 = audioplayer(AModded_Sig*.15, Fs,24);

Test1.play


pause(7)

Test2 = audioplayer(Carrier *.15, Fs, 24);

Test2.play

