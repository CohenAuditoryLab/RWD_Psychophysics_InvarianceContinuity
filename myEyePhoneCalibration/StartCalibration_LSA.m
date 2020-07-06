function [calibrationFile] = StartCalibration_LSA(aim)

scaleFactor = getStandardScaleFactor_AGifford();
% scaleFactor = [];
% scaleFactor = 1.0360;
% targetLevels = 60:5:85;
startMsg = 'Connect headphones from audio output to artifical ear. Press q to quit, or any other key to continue.';
start = input(startMsg,'s');

switch lower(start)
    case 'q'
        calibrationFile = [];
        return
    otherwise
%         filenames = [];
%         for tt = 1:length(targetLevels)
%             findFile = dir(['runCalibNoise-runID*-targetLevel' num2str(targetLevels(tt)) '.mat']);
%             if ~isempty(findFile)
%                 filenames = [filenames,{findFile.name}];
%             end
%         end
%         if length(filenames) < length(targetLevels)
        [filenames, targetLevels] = runCalib_LSA(aim,scaleFactor);
%         end
        d=[];voltages=[];levels=[];
        for i = 1:size(filenames,1)
            for j=1:size(filenames,2)
                d{i,j} = load(filenames{i,j});
                voltages(i,j,:) = permute(d{i,j}.gainValues,[3 2 1]);
                levels(i,j,:) = permute(d{i,j}.testedLevels,[3 2 1]);
            end
        end
        
%         calibrationMetadata = regexp(filenames{1},'runCalib-runID(?<dateCode>\d+T\d+)-itr\d*\.mat','names');
        
        
        calibratedVoltages = squeeze(median(voltages));
        calibratedVoltagesMean = squeeze(mean(voltages,1));
        calibratedVoltagesSD = squeeze(std(voltages));
        calibratedLevels = squeeze(mean(levels));
        calibratedLevelsSD = squeeze(std(levels));
        if strcmp(aim,'N')
            calibrationFile = struct('scaleFactor',scaleFactor,'calibratedamplitude',calibratedVoltages,...
            'calibratedVoltagesMean',calibratedVoltagesMean,'calibratedVoltagesSD',calibratedVoltagesSD,...
            'calibratedLevels',calibratedLevels,'calibratedLevelsSD',calibratedLevelsSD,...
            'targetLevels',targetLevels,'calibrationDate',date);
            save(['calibrationNoise-' date '.mat'],'calibrationFile');
        else
        cf = d{1}.cf;
%         targetLevel = d{1}.targetLevel;
        
        calibrationFile = struct('scaleFactor',scaleFactor,'calibratedamplitude',calibratedVoltages,...
            'calibratedVoltagesMean',calibratedVoltagesMean,'calibratedVoltagesSD',calibratedVoltagesSD,...
            'calibratedLevels',calibratedLevels,'calibratedLevelsSD',calibratedLevelsSD,'cf',cf,...
            'targetLevels',targetLevels,'calibrationDate',date);
            save(['calibration-' date '.mat'],'calibrationFile');
        end
        
end
end
