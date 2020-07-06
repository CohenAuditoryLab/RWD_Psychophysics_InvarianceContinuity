function plotCalibrationResults_LSA

% calibDataFolder = './psychophysicRoom/';
calibDataFolder = './';

listing_calibFiles = dir([calibDataFolder 'runCalib-*.mat']);
summary_calibFile = dir([calibDataFolder 'calibration-*.mat']);

calibTable = [];
for fileCounter = 1:size(listing_calibFiles,1)
    load([calibDataFolder listing_calibFiles(fileCounter).name]);
    
    nFreq = length(cf);
    for freqCounter = 1:nFreq
        curTable = table(cf(freqCounter),gainValues(freqCounter),testedLevels(freqCounter),round(testedLevels(freqCounter)),'VariableNames',{'frequency','gainValues','testedLevels','targetLevels'});
        calibTable = [calibTable; curTable];
    end
    
end

% noise
listing_calibFiles = dir([calibDataFolder 'runCalibNoise*.mat']);
summary_calibFile = dir([calibDataFolder 'calibrationNoise*.mat']);

calibTable = [];
for fileCounter = 1:size(listing_calibFiles,1)
    load([calibDataFolder listing_calibFiles(fileCounter).name]);
    
    curTable = table(gainValues,testedLevels,round(testedLevels),'VariableNames',{'gainValues','testedLevels','targetLevels'});
    calibTable = [calibTable; curTable];
    
end

% % to convert all calibration files into Adam's calibration data format
% tmp_table = unstack(grpstats(calibTable,{'frequency','targetLevels'},'median','DataVars','gainValues'),'median_gainValues','frequency');
% calibratedVoltages = tmp_table{:,3:6};
% tmp_table = unstack(grpstats(calibTable,{'frequency','targetLevels'},'mean','DataVars','gainValues'),'mean_gainValues','frequency');
% calibratedVoltagesMean = tmp_table{:,3:6};
% tmp_table = unstack(grpstats(calibTable,{'frequency','targetLevels'},'std','DataVars','gainValues'),'std_gainValues','frequency');
% calibratedVoltagesSD = tmp_table{:,3:6};
% tmp_table = unstack(grpstats(calibTable,{'frequency','targetLevels'},'mean','DataVars','testedLevels'),'mean_testedLevels','frequency');
% calibratedLevels = tmp_table{:,3:6};
% tmp_table = unstack(grpstats(calibTable,{'frequency','targetLevels'},'std','DataVars','testedLevels'),'std_testedLevels','frequency');
% calibratedLevelsSD = tmp_table{:,3:6};
% 
% calibrationFile = struct('calibratedamplitude',calibratedVoltages,...
%             'calibratedVoltagesMean',calibratedVoltagesMean,'calibratedVoltagesSD',calibratedVoltagesSD,...
%             'calibratedLevels',calibratedLevels,'calibratedLevelsSD',calibratedLevelsSD,'cf',cf,...
%             'targetLevels',targetLevels,'calibrationDate',date);
%         
%         save([calibDataFolder 'calibration-' date '.mat'],'calibrationFile');

load([calibDataFolder summary_calibFile.name])

g = gramm('x',calibTable.gainValues,'y',calibTable.testedLevels,'color',calibTable.frequency); %,'lightness',calibTable.targetLevels
g.geom_point;
% g.stat_summary('geom','line');
g.set_names('x','gain values (V)','y','tested levels (dB)','color','frequency'); %,'lightness','target levels'
g.set_title('ear phone calibration');
figure
g.draw();

curAx = gca;
for freqCounter = 1:nFreq
    hold on
    curColor = g.results.draw_data(freqCounter).color;
    plot(calibrationFile.calibratedVoltagesMean(:,freqCounter),calibrationFile.calibratedLevels(:,freqCounter),'Color',curColor,'LineWidth',2);
end