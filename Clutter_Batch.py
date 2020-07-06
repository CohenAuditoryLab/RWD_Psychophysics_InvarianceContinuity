# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:11:05 2020

Write interface that runs a stream lined version of MonkeyCall_Clutter

2020-06-18 Going to just get this to run with a slightly cleaned version of original code for
now then comeback and clean this up a lot.  I.e. I think this shell code should call an intermediate function
that then does the actual SFA stuff so options are easier to tweak and everything looks good.

Don't want to take forever with this though so only going to do a bit of clean for now and then do the rest next week or this weekend



@author: ronwd
"""

#Grab commonly used + definitely used in the code packages
import time
import numpy as np
import os
from scipy import stats, io
from pathlib import Path, PureWindowsPath
import matplotlib.pyplot as plt
import soundfile as sf 
import Calls_runSFA as rsfa

#Set up folders for stimulus pairs.  Will have to change for kilosort, but going to try some local runs now.
#Path is designed for everything to be unix based file names.  Start with this then convert to windows at the end

vocal_foldername = Path("C:/Users/ARL/.spyder-py3/SFA_Manuscript2020/HauserVocalizations50kHzSampling/Wav")

#Since we are loading this in once just set up as a windows path.
vocpairfile =  PureWindowsPath("C:\\Users\\ARL\\.spyder-py3\\SFA_Manuscript2020\\16-Jun-2020_pairs_list.mat")

#This will be used repeatedly so again leave as unix path and then switch to windows before loading
clutterfoldername =Path("C:/Users/ARL/.spyder-py3/SFA_Manuscript2020/16-Jun-2020_clutter/")
#Template for loading in each clutter chorus.
clutterfiletemplate = '_ClutterChorus_10calls.wav'

unpack_pairs = io.loadmat(vocpairfile)

all_pairs = unpack_pairs['listofpairs'] #weirdly loads in as a triple array but thats okay.  Maybe there is someway to squeeze this but haven't found it yet.

##This is just for a test.  But will need to use something like this to load in each pair.
##This works, good.
#file_to_open = vocal_foldername / all_pairs[0][0][0]
#
#test1, rate = sf.read(file_to_open)

total_stimuli = np.shape(all_pairs)[0] #get the number of pairs to set up for loop
#this hard coded for now but we will fix this later when we do the full clean later on, 20 rounds, and 9 SNR levels
#Need to do this weird 3rd x 1st x 2nd dimension thing because python
SFA_scores_batch = np.zeros([total_stimuli,20,9])
Baseline_scores_batch = np.zeros([total_stimuli,20,9])


t0=time.time()

#2020-06-19 get weird error for pair 7 where SVD doesn't converge.  Just skipping for now because that doesn't really make sense
#related, should figure out how to set up error catch block for this in case we get this job drop.

stimuli_ind_list = list(range(0,total_stimuli))
problem_stimuli = list([])


for cur_pair in stimuli_ind_list:  #run a quick batch here to check that is works then run on kilosort.  Works so lets port it over to kilosort
    
    print(cur_pair)
    try:
        
        [SFA_scores_batch[cur_pair,:,:] , Baseline_scores_batch[cur_pair,:,:] ]= rsfa.run_SFA(cur_pair,vocal_foldername,all_pairs[cur_pair],clutterfoldername, clutterfiletemplate)
    except:
        
        problem_stimuli.append(cur_pair)



t2run = time.time()-t0
print(t2run/60/60)



var_out = dict();

for i_vars in ('SFA_scores_batch','Baseline_scores_batch','all_pairs','problem_stimuli','stimuli_ind_list', 'unpack_pairs'):
    
   var_out[i_vars]=locals()[i_vars];
   
cwd = os.getcwd() 
   
io.savemat(cwd + '\\try1.mat', var_out)

'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

used_stimuli_ind_list = [i for i in stimuli_ind_list if i not in problem_stimuli]

##Post batch analysis.###########################

#dropping this in for plotting for now change later
snr_values = np.array([0.001, 0.05, 0.5, 1.0, 5, 10, 15, 25, 50]) 

#remove pairs that were not ran (or just leave in zeros)


#get average over iterations
score_avg = np.mean(SFA_scores_batch, axis = 1)

#remove pairs that were not ran (or just leave in zeros)

score_avg = np.delete(score_avg, problem_stimuli, 0)

#average across all ran pairs
score_avg_all = np.mean(score_avg, axis = 0)

#not the best to leave this here but just have it because below code generates a lot of plots and am currently modifying these images.
plt.close('all')
#plot change with SNR for all pairs
plt.figure()
plt.plot(snr_values, score_avg_all)
plt.xlabel('SNR')
plt.ylabel('Classification Accuracy')

#histogram of all classification values for each snr for all pairs
for snr_ind in range(0,snr_values.size-1):
    plt.figure()
    plt.hist(score_avg[:,snr_ind])
    plt.xlabel('Classification Accuracy')
    plt.ylabel('Counts (i.e. number of pairs')
    
#Get identity of each pair, build frequency plot matrix (i.e. times it was co vs co or grunt vs co etc.) and performance.
prefixes = list(['ha' ,'co', 'gt','sb'])
call_counter1 = np.zeros([1,4]) #using matrix mult to get the combo identity of each pair
call_counter2 = np.zeros([1,4]) #doing it this way since I don't want to have to call newaxis each time
combo_counter = np.zeros([4,4]) 
call_snr_avg = np.zeros([9,4,4]) #add the performance for each pair to one spot in the matrix then get average by dividing by combo_counter
    
for pair in used_stimuli_ind_list:
    #all_pairs as a weird array in array in array stucture from being a matlab cell array originally
    
    cur_pair_first = str(all_pairs[pair][0][0])
    cur_pair_second = str(all_pairs[pair][1][0])
    
    #create a vector which identifies which of the four calls that call is
    
    call_counter1[0,:] = np.array([cur_pair_first[0:2] == prefixes[0],cur_pair_first[0:2] == prefixes[1], \
                                 cur_pair_first[0:2] == prefixes[2], cur_pair_first[0:2] == prefixes[3]])
    
    
    call_counter2[0,:] = np.array([cur_pair_second[0:2] == prefixes[0],cur_pair_second[0:2] == prefixes[1], \
                                 cur_pair_second[0:2] == prefixes[2], cur_pair_second[0:2] == prefixes[3]])
    
    temp = np.matmul(call_counter1.T,call_counter2)
    
    combo_counter +=  temp
    
    
##Massively inelegant but for now just call the for loop again and do a double for loop with snr length since I am sick of trying to get basic matrix stuff to work with python for now    
    #Write this in the morning.

for snr_ind in range(0,snr_values.size):
    iteration = 0
    for pair in used_stimuli_ind_list:
            
            cur_pair_first = str(all_pairs[pair][0][0])
            cur_pair_second = str(all_pairs[pair][1][0])
            
            #create a vector which identifies which of the four calls that call is
            
            call_counter1[0,:] = np.array([cur_pair_first[0:2] == prefixes[0],cur_pair_first[0:2] == prefixes[1], \
                                         cur_pair_first[0:2] == prefixes[2], cur_pair_first[0:2] == prefixes[3]])
            
            
            call_counter2[0,:] = np.array([cur_pair_second[0:2] == prefixes[0],cur_pair_second[0:2] == prefixes[1], \
                                         cur_pair_second[0:2] == prefixes[2], cur_pair_second[0:2] == prefixes[3]])
            
            call_snr_avg[snr_ind,np.flatnonzero(call_counter1)[0],np.flatnonzero(call_counter2)[0]] += score_avg[iteration,snr_ind]
            iteration += 1
    plt.figure()
    plt.imshow(call_snr_avg[snr_ind,:,:]/combo_counter, extent=[0,1,0,1])
    plt.colorbar()
    plt.xticks([ 0.125, 0.375,  0.625, 0.875], prefixes)
    plt.yticks([ 0.875, 0.625, 0.375, 0.125], prefixes)
    plt.title(['Accuracy for SNR of ' + str(snr_values[snr_ind])])
        
plt.figure() #I think we need to transpose label spots...done
plt.imshow(combo_counter, extent=[0,1,0,1])
plt.xticks([ 0.125, 0.375,  0.625, 0.875], prefixes)
plt.yticks([ 0.875, 0.625, 0.375, 0.125], prefixes)
plt.title('Frequency of pairs')
plt.colorbar()



    