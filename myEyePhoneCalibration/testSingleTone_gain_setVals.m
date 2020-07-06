function [gain, level] = testSingleTone_gain_setVals(cf,V)
cf = 400:25:4000;
gain = zeros(numel(cf),1);
level = zeros(numel(cf),1);
targetLevel = 65; %db
%perFreqIteration = cell(numel(cf),2);
filterCoeff = load('postProcessFilter.mat');

for cfIdx = 1:length(cf)
    
    [data fs] = singleTone_calibration(cf(cfIdx),V(cfIdx));
    data = filter(filterCoeff.Hd,data-mean(data));
    testLevel = 20*log10(mean(sqrt(data.^2))/2e-5); % scale factor: 1 V/Pa
   
    
    fprintf('f: %d V:%.3f SPL: %.3f\n', cf(cfIdx),V(cfIdx),testLevel);
    level(cfIdx) = testLevel;
    gain(cfIdx) = V(cfIdx);
    %perFreqIteration(cfIdx,:) = {trialGains; trialLevels};
end
end