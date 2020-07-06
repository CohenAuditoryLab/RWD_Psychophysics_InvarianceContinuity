function [gain, level] = testNoiseGain_LSA(cf,targetLevel,scaleFactor)
gain = zeros(numel(cf),1);
level = zeros(numel(cf),1);
allTrialGains = cell(numel(cf),1);
allTrialLevels = cell(numel(cf),1);
% targetLevel = targetLevel/2;

% Lalitta - copy from rig calibration  
acc = 0.1;  % accuracy of calibration in dB SPL
% VUL = 1.22; % voltage upper limit
% ld = 0.1; % low delta for incrementing voltage;
% hd = 0.8; % high delta for incrementing voltage;
initialGain = 0.0001;

for cfId=1:length(cf)
    trialGains = [];
    trialLevels = [];
    
    if cfId > 1
        gainTest = gain(cfId-1,1);
    else
%         gainTest = 0.03;
        gainTest = initialGain; % minimum
    end
    
%     [data Fs] = singleToneCalib_AGifford(cf(cfId),gainTest);
    [data Fs] = singleToneCalib_LSA(cf(cfId),gainTest);
    
    data = (data-mean(data))*scaleFactor;
    testLevel = 20*log10(sqrt(mean(data.^2))/2e-5); % scale factor: 1 V/Pa
    trialGains = [trialGains gainTest];
    trialLevels = [trialLevels testLevel];
    
    trialCount = 1;
    while abs(testLevel-targetLevel)>acc
        if trialCount<30
            if abs(testLevel-targetLevel)<3
                gainTest = gainTest*10^((-(testLevel-targetLevel)/20)*(1+0.0005*randn(1,1)));
            else
                gainTest = gainTest*10^((-(testLevel-targetLevel)/20)*(1+0.001*randn(1,1)));
            end
            
%             if gainTest>1.4
%                 gainTest = gainTest*0.1;
%             end
%         elseif trialCount>=30 && (max(trialLevels)<targetLevel+5)
%             gainTest = 1.4;
%         elseif trialCount>=30 && max(trialLevels)>=targetLevel
%             gainTest = interp1(targetLevel,trialLevels,trialGains,'pchip');
%             if gainTest>1.4
%                 gainTest = gainTest*0.1;
%             end
        end
        
        fprintf('gainTest: %d\n trial level: %d\n\n',gainTest,testLevel);
  %     [data Fs] = singleToneCalib_AGifford(cf(cfId),gainTest);
        [data Fs] = singleToneCalib_LSA(cf(cfId),gainTest);
        data = (data-mean(data))*scaleFactor;
        testLevel = 20*log10(sqrt(mean(data.^2))/2e-5);
        
        trialGains = [trialGains gainTest];
        trialLevels = [trialLevels testLevel];
        
        trialCount = trialCount+1;
    end
    
    % Lalitta - copy from rig calibration
%     diffAmp = targetLevel - testLevel;
%     while abs(diffAmp) > acc
%         if diffAmp > 1
%             gainTest = gainTest * (1+(diffAmp/5)*hd);
%         elseif diffAmp > 0 && diffAmp < 1
%             gainTest = gainTest * (1+(diffAmp/5)*ld);
%         elseif diffAmp < -1
%              gainTest = gainTest / (1+abs(diffAmp/5)*hd);
%         elseif diffAmp < 0 && diffAmp > -1
%             gainTest = gainTest / (1+(diffAmp/5)*ld);
%         end
%         if gainTest > VUL
%             disp('***** Voltage set is > 1.22 V *****')
%             diffAmp = 0;
%         elseif gainTest < initialGain
%             disp('***** Voltage set is < 0.005 V *****')
%             diffAmp = 0;
%         else
%             fprintf('gainTest: %d\n trial level: %d\n\n',gainTest,testLevel);
%             [data Fs] = singleToneCalib_AGifford(cf(cfId),gainTest);
%             data = (data-mean(data))*scaleFactor;
%             testLevel = 20*log10(sqrt(mean(data.^2))/2e-5);
%             
%             trialGains = [trialGains gainTest];
%             trialLevels = [trialLevels testLevel];
%             
%             trialCount = trialCount+1;
%         end
%     end
%     
    fprintf('f: %d V:%.3f SPL: %.3f\n', cf(cfId),gainTest,testLevel);
    level(cfId) = testLevel;
    gain(cfId) = gainTest;
    allTrialGains{cfId} = trialGains;
    allTrialLevels{cfId} = trialLevels;
end