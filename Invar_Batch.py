# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:09:02 2020

Invar batch interface code


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
import Invar_runSFA as rsfa

'Update 2020-08-26: need to edit this to run with Invar stuff, right now this is just a c+p of clutter'
'Update folders and add 5 rounds thing to Invar_runsSFA'
'Will update folders after moving to kilosort but this is finally ready to go'

##############IMPORTANT NEED TO PUSH OVER THE NEW STIMULUS AND NOISE FILES TO KILOSORT

#Set up folders for stimulus pairs.  Will have to change for kilosort, but going to try some local runs now.
#Path is designed for everything to be unix based file names.  Start with this then convert to windows at the end
vocal_foldername = Path("C:/Users/ARL/.spyder-py3/SFA_Manuscript2020/HauserVocalizations50kHzSampling/Wav")

#Since we are loading this in once just set up as a windows path.
vocpairfile =  PureWindowsPath("C:\\Users\\ARL\\.spyder-py3\\SFA_Manuscript2020\\21-Jul-2020_pairs_list.mat")

#NEED TO UPDATE BELOW##
trainingvocfile = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_PostCOSYNEAPP\\Invar_Stim_juneon\\18-Aug-2020_traininggroups.mat")

unpack_pairs = io.loadmat(vocpairfile)

    
unpack_training = io.loadmat(trainingvocfile)
   
   
#Template for loading in each clutter chorus.

all_pairs = unpack_pairs['listofpairs']
all_training = unpack_training['Training_Vocals']

total_stimuli = np.shape(all_pairs)[0] #get the number of pairs to set up for loop
#this hard coded for now but we will fix this later when we do the full clean later on, 20 rounds, and 9 SNR levels
#Need to do this weird 3rd x 1st x 2nd dimension thing because python

#Update 2020-08-04 new shape for these vectors based on new design 5 rounds with 8 SNR values.
#Also added null variables
SFA_scores_batch = np.zeros([total_stimuli,5,3])
Baseline_scores_batch = np.zeros([total_stimuli,5,3])
SFA_scores_batch_null = np.zeros([total_stimuli,5,3])
Baseline_scores_batch_null = np.zeros([total_stimuli,5,3])

t0=time.time()

#2020-06-19 get weird error for pair 7 where SVD doesn't converge.  Just skipping for now because that doesn't really make sense
#related, should figure out how to set up error catch block for this in case we get this job drop.

stimuli_ind_list = list(range(0,total_stimuli))
problem_stimuli = list([])
run_nulls = False

for cur_pair in stimuli_ind_list:  #run a quick batch here to check that is works then run on kilosort.  Works so lets port it over to kilosort
    
    print(cur_pair)
    try:
        
        [SFA_scores_batch[cur_pair,:,:] , Baseline_scores_batch[cur_pair,:,:] ]= rsfa.run_SFA(vocal_foldername,all_pairs[cur_pair],all_training[cur_pair],run_nulls)
    except:
        
        problem_stimuli.append(cur_pair)



t2run = time.time()-t0
print(t2run/60/60)

t1 = time.time()

run_null = True

for cur_pair in stimuli_ind_list:  #run a quick batch here to check that is works then run on kilosort.  Works so lets port it over to kilosort
    
    print(cur_pair)
    try:
        
        [SFA_scores_batch_null[cur_pair,:,:] , Baseline_scores_batch_null[cur_pair,:,:] ]= rsfa.run_SFA(vocal_foldername,all_pairs[cur_pair],all_training[cur_pair],run_nulls)
    except:
        
        problem_stimuli.append(cur_pair)

var_out = dict();

for i_vars in ('SFA_scores_batch','Baseline_scores_batch','SFA_scores_batch_null','Baseline_scores_batch_null','all_pairs','problem_stimuli','stimuli_ind_list', 'unpack_pairs'):
    
   var_out[i_vars]=locals()[i_vars];
   
cwd = os.getcwd() 
   
io.savemat(cwd + '\\try1_newcode_AUG.mat', var_out)
