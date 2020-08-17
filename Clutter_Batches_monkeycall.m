%% Clutter full batch setup
%Create batches of pairs vocalizations and clutters to be analyzed
%Pull across the 4 different groups using 20 potential samples.
%For now grab samples randomly and take a subset of those (otherwise it
%will take forever to run).  To start going to just do 100 samples with 10
%iterations to pin down variance is only 1000 runs.  Each run takes
%approximately 35.5 seconds so this shouldn't take too long.

%% Pull 19 random vocalization from the 4 different catagories


categories = 4;
samples = 19; %2020-06-16 apparently there are only 19 harmonic arches in the wav file


callprefix = {'gt' ,'sb', 'ha' ,'co'};%There are set prefixes before each of the calls

%Self note: 2020-06-15 there is on coo that is not a coo that is
%mislabeled.  Make sure you check for this.  Update the wav file is
%properly named so don't have to worry about this.

listofvocs = cell(samples,categories); %this will be the matrix of all the pulled vocalizations 
usedvocs_inds = zeros(samples,categories); %this will save the random indices that were used
%(inds are relative to the vocalization category (e.g. 1 means using the
%first grunt while 5 means using the 5th grunt, not the 5th overall
%vocalization)

vocpath = 'C:\Users\ronwd\.spyder-py3\SFA_PostCOSYNEAPP-master\Monkey_Calls\HauserVocalizations\Monkey_Wav_50K\Wav\';
allvoc = dir(vocpath);
allvoc = {allvoc.name};
allvoc = allvoc(3:end); %the first and second outputs from name are always . and ..


for i = 1:categories
    curr_vocs = allvoc(find(contains(allvoc,callprefix{i})));

    usedvocs_inds(:,i) = randsample(length(curr_vocs),samples);
    
    listofvocs(:,i) = curr_vocs(usedvocs_inds(:,i));
    
end

save([date '_vocalization_list'], 'listofvocs', 'usedvocs_inds')

%% Pull 100 of the random pairs from above

pairs_to_use = 100;
%switch listofvocs into a column cell array from a matrix

listofvocs = reshape(listofvocs,samples*categories,1);

%2020-06-16 there is probably a better way to do this but going to get a
%random grouping of pairs without replacement and without allowing same
%matches using an ind2sub idea

[row, col] = ind2sub([length(listofvocs) length(listofvocs)], randsample(length(listofvocs)*length(listofvocs),pairs_to_use)); %grab 100 random points on the page

if any(row == col) %don't want comparisons with self
   
    row(row==col) = randsample(length(listofvocs),1); %just change the rows
    
       while length(unique([row col], 'rows')) ~= pairs_to_use %check that have all unique pairs
           
           %redo sample and check again that there are no row==col
           
           if any(row == col)
               
                row(row==col) = randsample(length(listofvocs),1); %just change the rows
               
           end
           
       end
       
end

pair_inds = [row,col]; %use the pairs of vocalizations from listofvocs

listofpairs = [listofvocs(pair_inds(:,1)) listofvocs(pair_inds(:,2))];

save([date '_pairs_list'], 'pair_inds', 'listofpairs')

%% Generate clutter chorus for each pair making sure the pair isn't included:

save_path = [cd '\' date '_clutter']; %just do this for now and then put things in a folder as you need.

for pairs = 1:length(listofpairs) % for all pairs
    
    cd 'C:\Users\ronwd\Desktop\MATLAB\PsychoPhysicsWithYale\stimulus_generation_SFA_manu'
    
    voc1_name = listofpairs(pairs,1);
    voc2_name =  listofpairs(pairs,2);
    
    Signal_files = [voc1_name voc2_name];
    
    Clutter_files = listofvocs;
    shared_inds = ismember(Clutter_files, Signal_files); %remove the pair from the list of all vocalizations
    Clutter_files(shared_inds)= []; 
    
    
    


% Grab wav files and other information for clutter files

Stim_Clutter = cell(length(Clutter_files),1);

for i =1:length(Clutter_files)
  
    [Full_stim ,Fs] = audioread([vocpath Clutter_files{i}]);
    Full_Duration = length(Full_stim)/Fs;
    
    Stim_Clutter{i}.Full_stim = Full_stim;
    Stim_Clutter{i}.Fs = Fs;
    Stim_Clutter{i}.Full_Duration = Full_Duration;
    
    
end

%Actually make choruses %editted function so save_path is in folder in cd and each
%file is saved with the pair index leading.

[ychorus, randomgrabs] = make_chorus_calls_RWD(Stim_Clutter, save_path, pairs); %function to create choruses of calls, Outer_Path is folder to save file in.


end