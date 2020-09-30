# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:12:04 2020

Code to plot results after running Clutter_Batch

2020-07-09 needs to be edited to work again, some issues with loaded variables...

2020-08-11 still need to fix this, just plotting some basics now and commenting out what doesn't work or isn't relevant

@author: ARL
"""
import numpy as np
import os
from scipy import stats, io
from pathlib import Path, PureWindowsPath
import matplotlib.pyplot as plt

#plotting is mess up again, but just come back and fix this later
#making this a new file now, leaving here commented out for posterity
cwd = os.getcwd()

# unpack = io.loadmat(cwd + '\\try1.mat')

SFA_scores_batch = unpack['SFA_scores_batch']
Baseline_scores_batch = unpack['Baseline_scores_batch']
all_pairs = unpack['all_pairs']
problem_stimuli = unpack['problem_stimuli']
stimuli_ind_list = unpack['stimuli_ind_list']
unpack_pairs = unpack['unpack_pairs']

# #looks like we need to fix this line.
# used_stimuli_ind_list = [i for i in stimuli_ind_list if i not in problem_stimuli]

# ##Post batch analysis.###########################

# #dropping this in for plotting for now change later
# snr_values = np.array([0.001, 0.05, 0.5, 1.0, 5, 10, 15, 25, 50]) #for try 1
# #snr_values = np.array([0.0001, 0.001, 0.05, 0.1, 0.25,  0.5, 0.75, 1.0, 5]) #for try 2

# #remove pairs that were not ran (or just leave in zeros)

#get SNR values and take log

snr_values = np.array([1e-7, 1e-5, 1e-3, .1, 1.0, 10.0, 100.0, 1000.0])

#get average over iterations
score_avg = np.mean(SFA_scores_batch_null, axis = 1)

# #remove pairs that were not ran (or just leave in zeros)

# score_avg = np.delete(score_avg, problem_stimuli, 0)

#average across all ran pairs
score_avg_all = np.mean(score_avg, axis = 0)

#not the best to leave this here but just have it because below code generates a lot of plots and am currently modifying these images.
plt.close('all')
#plot change with SNR for all pairs
plt.figure()
plt.plot(np.log10(snr_values), score_avg_all)
plt.xlabel('SNR')
plt.ylabel('Classification Accuracy')
#plt.plot()

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

#for some reason this code is not working with loaded data...trouble shoot
    
for pair in used_stimuli_ind_list:
    #all_pairs as a weird array in array in array stucture from being a matlab cell array originally
    
    cur_pair_first = all_pairs[pair][0][0][0] #looks like if we load in all_pairs it doesnt need to have this be a string
    cur_pair_second = all_pairs[pair][1][0][0] #also if you load in all_pairs it gets another weird dimension
    
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
            
            cur_pair_first = all_pairs[pair][0][0][0]
            cur_pair_second = all_pairs[pair][1][0][0]
            
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



    