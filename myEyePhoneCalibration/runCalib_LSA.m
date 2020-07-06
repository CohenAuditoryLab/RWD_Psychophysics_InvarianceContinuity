function [fileNames targetLevels] = runCalib_LSA(aim,scaleFactor)
nReps = 1;

pause(5)

switch num2str(aim)
    case '1'
        targetLevels = 60:1:90; % db
        cf = [250 500 1000 2000]; % hz
%         cf = 1000*2.^((0:8)/48); % hz
%         cf = 1000;
     case 'N'
        targetLevels = 75; % db
    otherwise
        error('Invalid aim input.')
end
fileNames = cell(nReps,length(targetLevels));
timeStamp = datestr(now,30);

for j=1:nReps
    for k=1:length(targetLevels)
        targetLevel = targetLevels(k);
        if strcmp(aim,'N')
            % calibrate noise
            [gainValues, testedLevels, trialGainValues, trialTestedValues] = noiseCalib_LSA(targetLevel,scaleFactor);
            fileName = ['runCalibNoise-runID' timeStamp '-itr' num2str(j) '-targetLevel' num2str(targetLevel) '.mat']; 
            save(fileName,'gainValues','testedLevels','trialGainValues','trialTestedValues','targetLevel','scaleFactor');
        else
            [gainValues, testedLevels, trialGainValues, trialTestedValues] = testSingleToneGain_LSA(cf,targetLevel,scaleFactor);
            fileName = ['runCalib-runID' timeStamp '-itr' num2str(j) '-targetLevel' num2str(targetLevel) '.mat']; 
            save(fileName,'gainValues','testedLevels','trialGainValues','trialTestedValues','cf','targetLevel','scaleFactor');
        end
        fileNames{j,k} = fileName;
    end
end

