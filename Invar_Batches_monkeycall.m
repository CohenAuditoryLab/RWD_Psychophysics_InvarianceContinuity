%% Invar full batch setup
%Also for now just grabbing random groups as doing the whole space is not
%feasible.  Implement most of this this evening and then ask Yale about it
%tomorrow.

%% Pull 20 random coos and grunts
%For now using coos and grunts to tie to other paper by Yale.

%2020-09-10 trying co vs ha and gt vs sb next


categories = 2;
samples = 19; %Just pull 20 vocals for the sake of it being an even number. ha only has 19


callprefix = {'ha' ,'sb'};%There are set prefixes before each of the call, just using grunts and coos this time

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

save([date '_invar_vocalization_list'], 'listofvocs', 'usedvocs_inds')

%% Pull 100 of the random pairs from above
%2020-08-17, current logic is to pull a few pairs for use in testing and
%then use noise/clutter generating code to make training data sets.
%Probably use 10 vocalizations for training, since that is easier to
%implement, and do the three conditions 10 of one 5 and 5 10 of the other.
%Since doing without noise can use 100 pairs again.  Each pair will be run
%3 times for the different training sets and 5 times for the potential
%variance.

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

%% Get sets of vocalizations for training set
%2020-08-17 taking a quick break, but come back to thinking through this
%section.

save_path = [cd '\' date '_traininggroups']; %just do this for now and then put things in a folder as you need.
Training_sets = 3; %Set number of training groups
Training_form =[5 5; 10 0; 0 10]; %Set up "form" of training group.  In our case, as there are two categories of vocalizations we will simply do even split, all of one, all of another
%^^^^^^^
%IMPORTANT TO NOTE THAT ORDER MATTERS, DOUBLE CHECK FORM MATCHES THE ORDER
%OF PREFIXES IN CALLPREFIX***********************************************

Training_Vocals = cell(length(listofpairs),Training_sets);

%Don't love all the nested for loops but for now its fine.  Only run this
%once to set up data set and then move on

for pairs = 1:length(listofpairs) % for all pairs
    
    %note needed anymore% cd 'C:\Users\ronwd\Desktop\MATLAB\PsychoPhysicsWithYale\stimulus_generation_SFA_manu'
    
    voc1_name = listofpairs(pairs,1);
    voc2_name =  listofpairs(pairs,2);
    
    Signal_files = [voc1_name voc2_name];
    
    Other_files = listofvocs;
    shared_inds = ismember(Other_files, Signal_files); %remove the pair from the list of all vocalizations
    Other_files(shared_inds)= []; 
    
    
    


% Grab file names for each training set

for i= 1:Training_sets
        
        num_cats_used = sum(Training_form(i,:)>0); %get number of vocal categories used
        
        inds_of_cats = find(Training_form(i,:)>0); %get indecies for used categories
        
        
        
        
        if num_cats_used>1 %if using more than one vocalization, need to make sure get correct number of samples of each vocalization
            
                
            for ii = 1:num_cats_used %don't love nested for loop, but can't think of a more elegant solution right now, and this doesn't take very long to run
                
                temp_vocs = Other_files(find(contains(Other_files, callprefix{inds_of_cats(ii)}))); %get list of vocals only from that category.
                
                usedvocs_inds = randsample(length(temp_vocs),Training_form(i,inds_of_cats(ii))); %get random vocals from that list equal to the training form number
                
                Training_Vocals{pairs,i} = [Training_Vocals{pairs,i}; temp_vocs(usedvocs_inds)]; %concatante all of the vocals in a training set together.  Note this means the training set vocals will start separated by glass, but the SFA code itself will shuffle them.
                
                
            end
            
        else %if only using one doesn't matter
            
            curr_vocs = Other_files(find(contains(Other_files,callprefix{inds_of_cats}))); %only take the prefix (i.e. use that vocal category) if going to have more than zeros samples
            
            usedvocs_inds = randsample(length(curr_vocs),Training_form(i, inds_of_cats)); %Take whatever the nonzero number is of Training_form
            
            Training_Vocals{pairs,i} = curr_vocs(usedvocs_inds); %Don't need to concatenate since only one category is used
            
        end
        
        
    

end

end

%I think we can just save this, may need more information for more complex
%things later, but at least that information is recoverable

save(save_path, 'Training_Vocals') %save the sets of Training_Vocals