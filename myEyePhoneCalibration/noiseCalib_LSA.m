function [gain, level, trialGains, trialLevels] = noiseCalib_LSA(targetLevel,scaleFactor)

calibNoiseFileName = 'calibNoise.mat';

acc = 0.01;
initialGain = 1;
trialGains = [];
trialLevels = [];

gainTest = initialGain;

if ~exist(calibNoiseFileName,'file')
    singleNoiseCalibFilterGen_LSA(calibNoiseFileName);
end
load(calibNoiseFileName)

testLevel = singleNoiseCalib_LSA(gainTest,scaleFactor,CalibData.hinv);
trialGains = [trialGains gainTest];
trialLevels = [trialLevels testLevel];


trialCount = 1;
while abs(testLevel-targetLevel) > acc
    if abs(testLevel-targetLevel) < 3
        gainTest = gainTest*10^((-(testLevel-targetLevel)/20)*(1+0.0005*randn(1,1)));
    else
        gainTest = gainTest*10^((-(testLevel-targetLevel)/20)*(1+0.001*randn(1,1)));
    end

    fprintf('gainTest: %d\n trial level: %d\n\n',gainTest,testLevel);
    
    testLevel = singleNoiseCalib_LSA(gainTest,scaleFactor,CalibData.hinv);

    trialGains = [trialGains gainTest];
    trialLevels = [trialLevels testLevel];
    
    gain = gainTest;
    level = testLevel;
    
    trialCount = trialCount+1;
end


disp('hey')