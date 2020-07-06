function scaleFactor = getStandardScaleFactor_AGifford()

Fs = 44100;
recObj = audiorecorder(Fs,24,1,0);

startMsg = 'Turn on standard tone. Press q to quit, or any other key to continue.';
start = input(startMsg,'s');

switch lower(start)
    case 'q'
        scaleFactor = [];
        return
    otherwise
        fprintf('\nRecording standard tone.\n');
        
        recordblocking(recObj,6)
        fprintf('Recording complete. Calculating scale factor.\n');
        
        lineIn = getaudiodata(recObj);
        
        numSamps = length(lineIn);
        cutOff = round(numSamps/3);
        
        lineIn = lineIn(cutOff+1:end-cutOff);
        lineIn = lineIn-mean(lineIn);
        
        Vrms = sqrt(mean(lineIn.^2));
        scaleFactor = 1/Vrms;
        
        fprintf('Finished. Scale factor: %d. Turn off tone.\n',scaleFactor);
end