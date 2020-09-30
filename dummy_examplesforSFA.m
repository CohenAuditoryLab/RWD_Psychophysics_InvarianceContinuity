%% Script to make dummy examples to display how SFA works.

%1 is a one second example, 2 is a longer example
%b or other letters indicate unique frequncy used
%simplest means it is just the first term
%% Take inspiration from Bellec and 2002 original paper and use their toy example

%They have 5 element input where the slowest signal is a sine wave of a set
%frequence.
fs = 50000; %set sampling frequency to the same as the monkey vocals 50k
f0 = 1; 
t=0:1/fs:1; %one second of sound since vocalizations are pretty short overall and to prevent memory issues
alpha = 1; %just have this as one for now

%now follow their equations

x1 = sin(2*pi*f0*t) + alpha * cos(11*2*pi*f0*t).^2;
x2 = cos(11*2*pi*f0*t);
x3 = x1 .* x1;
x4 = x1 .* x2;
x5 = x2 .* x2;

X =[x1;x2;x3;x4;x5];

% %visualize real quick with f0 as 1 and a three second window
% %now put back to f0 as 400 and one second window
% 
% for i = 1:5
%     figure
%     plot(t,X(i,:))
%     axis([0 3 -5 5])
%     
% end

%combine all signals together, treating them like a weird cluster of
%voices/instruments

%there was clipping during the wav file generation.  try scaling down.
%scaling down by 10 seems to prevent any clipping.

signal = sum(X,1)/10;
size(signal)

%make into a wav file
nbits = 24;
outName = 'dummyexamplesimplestb';
saveDir = 'C:\Users\ronwd\OneDrive\Documents\GitHub\SFA_PostCOSYNEAPP';

cd(saveDir)
audiowrite([outName '.wav'],signal,fs,'BitsPerSample',nbits)

