# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:56:35 2020

Local plotting of results from kilosort

2020-08-27 getting weird results with null so think this is the wrong one, since I checked nulls on kilosort not too long ago and it looked correct.
Check if wrong file.  It is the wrong file, the null file apparently didn't save so I am rerunning it concurrently with the invar stuff now

@author: ronwd
"""

import numpy as np
import matplotlib.pyplot as plt 
import SFA_Tools.SFA_Sets as s
from SFA_Tools.SFA_Func import *
from pathlib import Path, PureWindowsPath
import os
from scipy import stats, io
from collections import defaultdict


#Set file

results_file = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_results_fromkilo\\try1_newcode_AUG.mat")
#New range 2020-08-04
snr_values = np.array([1e-7, 1e-5, 1e-3, .1, 1.0, 10.0, 100.0, 1000.0]) 

unpack_data = io.loadmat(results_file)
unpack_pairs = unpack_data['stimuli_ind_list']
all_pairs = unpack_data['all_pairs']

Baseline_scores_batch = unpack_data['Baseline_scores_batch']
SFA_scores_batch = unpack_data['SFA_scores_batch']

null_file = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_results_fromkilo\\try1_nulls_newcode_AUG.mat")
unpack_data = io.loadmat(null_file)
SFA_scores_batch_null = unpack_data['SFA_scores_batch_null']
Baseline_scores_batch_null = unpack_data['Baseline_scores_batch_null']


#grab avg and std for 5 iterations of each of the vocalizations, then repeat over all vocalizations
avg_per_vocal_SFA = np.mean(SFA_scores_batch, axis = 1)
std_per_vocal_SFA = np.std(SFA_scores_batch, axis = 1)
avg_per_vocal_SFAnull = np.mean(SFA_scores_batch_null, axis = 1)
std_per_vocal_SFAnull = np.std(SFA_scores_batch_null, axis = 1)

avg_per_vocal_Baseline = np.mean(Baseline_scores_batch, axis = 1)
std_per_vocal_Baseline = np.std(Baseline_scores_batch, axis = 1)
avg_per_vocal_Baselinenull =np.mean(Baseline_scores_batch_null, axis = 1)
std_per_vocal_Baselinenull =np.std(Baseline_scores_batch_null, axis = 1)

#Get overall average

ov_avg_SFA = np.mean(avg_per_vocal_SFA,axis = 0)
ov_std_SFA = np.std(avg_per_vocal_SFA,axis = 0)
ov_avg_SFAnull = np.mean(avg_per_vocal_SFAnull,axis = 0)
ov_std_SFAnull = np.std(avg_per_vocal_SFAnull,axis = 0)


ov_avg_Baseline = np.mean(avg_per_vocal_Baseline,axis = 0)
ov_std_Baseline = np.std(avg_per_vocal_Baseline,axis = 0)
ov_avg_Baselinenull = np.mean(avg_per_vocal_Baselinenull,axis = 0)
ov_std_Baselinenull = np.std(avg_per_vocal_Baselinenull,axis = 0)


#plot some random examples

examples = 5
rnd_vocal = np.random.choice(SFA_scores_batch.shape[0],examples,replace = False)
for i in range(0,examples):
    plt.figure()
    plt.plot(np.log10(snr_values),avg_per_vocal_SFA[rnd_vocal[i],:],'b.')
    plt.plot(np.log10(snr_values),avg_per_vocal_SFA[rnd_vocal[i],:] + std_per_vocal_SFA[rnd_vocal[i],:],'b2')
    plt.plot(np.log10(snr_values),avg_per_vocal_SFA[rnd_vocal[i],:] -1.0*std_per_vocal_SFA[rnd_vocal[i],:],'b1')
      
    plt.plot(np.log10(snr_values),avg_per_vocal_Baseline[rnd_vocal[i],:],'r.')
    plt.plot(np.log10(snr_values),avg_per_vocal_Baseline[rnd_vocal[i],:] + std_per_vocal_SFAnull[rnd_vocal[i],:],'r2')
    plt.plot(np.log10(snr_values),avg_per_vocal_Baseline[rnd_vocal[i],:] - 1.0*std_per_vocal_SFAnull[rnd_vocal[i],:],'r1')
    #plt.plot(np.log10(snr_values),avg_per_vocal_SFAnull[rnd_vocal[i],:],'r.')
    #plt.plot(np.log10(snr_values),avg_per_vocal_SFAnull[rnd_vocal[i],:] + std_per_vocal_SFAnull[rnd_vocal[i],:],'r2')
    #plt.plot(np.log10(snr_values),avg_per_vocal_SFAnull[rnd_vocal[i],:] - 1.0*std_per_vocal_SFAnull[rnd_vocal[i],:],'r1')
    plt.xlabel('Log_10 SNR')
    plt.ylabel('Classification Accuracy')
    plt.title([all_pairs[rnd_vocal[i]][0][0] + ' ' + all_pairs[rnd_vocal[i]][1][0]])
#plot overall average

plt.figure()
plt.plot(np.log10(snr_values),ov_avg_SFA[:],'b.')
plt.plot(np.log10(snr_values),ov_avg_SFA[:] + ov_std_SFA[:],'b2')
plt.plot(np.log10(snr_values),ov_avg_SFA[:] -1.0*ov_std_SFA[:],'b1')

plt.plot(np.log10(snr_values),ov_avg_Baseline[:],'r.')
plt.plot(np.log10(snr_values),ov_avg_Baseline[:] + ov_std_SFAnull[:],'r2')
plt.plot(np.log10(snr_values),ov_avg_Baseline[:] - 1.0*ov_std_SFAnull[:],'r1')

# plt.plot(np.log10(snr_values),ov_avg_SFAnull[:],'r.')
# plt.plot(np.log10(snr_values),ov_avg_SFAnull[:] + ov_std_SFAnull[:],'r2')
# plt.plot(np.log10(snr_values),ov_avg_SFAnull[:] - 1.0*ov_std_SFAnull[:],'r1')

plt.xlabel('Log_10 SNR')
plt.ylabel('Classification Accuracy')
plt.title('Overall Average')

#################################################
#plot based on the call identity
#################################################
plt.close('all')
#Get identity of each pair, build frequency plot matrix (i.e. times it was co vs co or grunt vs co etc.) and performance.
prefixes = list(['ha' ,'co', 'gt','sb'])
call_counter1 = np.zeros([1,4]) #using matrix mult to get the combo identity of each pair
call_counter2 = np.zeros([1,4]) #doing it this way since I don't want to have to call newaxis each time
combo_counter = np.zeros([4,4]) 
call_snr_avg = np.zeros([8,4,4]) #add the performance for each pair to one spot in the matrix then get average by dividing by combo_counter

#get the number of each vocalization category
for pair in range(0,unpack_pairs.shape[1]):
    cur_pair_first = all_pairs[pair][0][0]
    cur_pair_second = all_pairs[pair][1][0]
    
    #create a vector which identifies which of the four calls that call is
    
    call_counter1[0,:] = np.array([cur_pair_first[0:2] == prefixes[0],cur_pair_first[0:2] == prefixes[1], \
                                 cur_pair_first[0:2] == prefixes[2], cur_pair_first[0:2] == prefixes[3]])
    
    
    call_counter2[0,:] = np.array([cur_pair_second[0:2] == prefixes[0],cur_pair_second[0:2] == prefixes[1], \
                                   cur_pair_second[0:2] == prefixes[2], cur_pair_second[0:2] == prefixes[3]])
        
        
    temp = np.matmul(call_counter1.T,call_counter2)
    
    combo_counter +=  temp   

#"fold" Diagonals of combo_counter as order doesn't matter.  Not sure if there is a more efficient way to do this but just going to do it this way for now.

cc_fold = np.triu(combo_counter).T + np.tril(combo_counter) #add upper tri transposed to lower tri
np.fill_diagonal(cc_fold, np.diag(combo_counter)) #reset the main diagonal to be what it actually is.
cc_fold[cc_fold<1]=-1 #set zeros equal to -1 to prevent division by zero


#Now get the average classification performance

for snr_ind in range(0,snr_values.size):
    iteration = 0
    for pair in range(0,unpack_pairs.shape[1]):
        
        cur_pair_first = all_pairs[pair][0][0]
        cur_pair_second = all_pairs[pair][1][0]
        
        #create a vector which identifies which of the four calls that call is
        
        call_counter1[0,:] = np.array([cur_pair_first[0:2] == prefixes[0],cur_pair_first[0:2] == prefixes[1], \
                                     cur_pair_first[0:2] == prefixes[2], cur_pair_first[0:2] == prefixes[3]])
        
        
        call_counter2[0,:] = np.array([cur_pair_second[0:2] == prefixes[0],cur_pair_second[0:2] == prefixes[1], \
                                       cur_pair_second[0:2] == prefixes[2], cur_pair_second[0:2] == prefixes[3]])
            
        call_snr_avg[snr_ind,np.flatnonzero(call_counter1)[0],np.flatnonzero(call_counter2)[0]] += avg_per_vocal_SFA[iteration,snr_ind]
            
        iteration += 1

#play same trick with folding diagonals as above.  Do it for each snr value

    csnra_fold =  np.triu(call_snr_avg[snr_ind,:,:]).T + np.tril(call_snr_avg[snr_ind,:,:]) #add upper tri transposed to lower tri
    np.fill_diagonal(csnra_fold, np.diag(call_snr_avg[snr_ind,:,:])) #reset the main diagonal to be what it actually is.          
                
 
    plt.figure()
    plt.imshow(csnra_fold/cc_fold, extent=[0,1,0,1])
    plt.clim(0.5, 1)
    plt.colorbar()
    plt.xticks([ 0.125, 0.375,  0.625, 0.875], prefixes)
    plt.yticks([ 0.875, 0.625, 0.375, 0.125], prefixes)
    plt.title(['Accuracy for SNR of ' + str(snr_values[snr_ind])])
        
plt.figure() #I think we need to transpose label spots...done
plt.imshow(cc_fold, extent=[0,1,0,1])
plt.xticks([ 0.125, 0.375,  0.625, 0.875], prefixes)
plt.yticks([ 0.875, 0.625, 0.375, 0.125], prefixes)
plt.title('Frequency of pairs')
plt.colorbar()

## Invar plotting
#Just manually do %reset
#2020-09-08 made same mistake with Null data so that is rerunning now

# import numpy as np
# import matplotlib.pyplot as plt 
# import SFA_Tools.SFA_Sets as s
# from SFA_Tools.SFA_Func import *
# from pathlib import Path, PureWindowsPath
# import os
# from scipy import stats, io

plt.close('all') #Just in case
results_file = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_results_fromkilo\\try3_INVAR_SEP.mat")

unpack_data = io.loadmat(results_file)
unpack_pairs = unpack_data['stimuli_ind_list']
all_pairs = unpack_data['all_pairs']

Baseline_scores_batch = unpack_data['Baseline_scores_batch']
SFA_scores_batch = unpack_data['SFA_scores_batch']

#take average for each training set for each vocal across the 5 repeats#2020-09-08 this is not working conventionally, so just going to hand code and come back to it

avg_per_vocal_SFA= np.mean(SFA_scores_batch, axis = 1)
std_per_vocal_SFA = np.std(SFA_scores_batch, axis = 1)

avg_per_vocal_Baseline = np.mean(Baseline_scores_batch, axis = 1)
std_per_vocal_Baseline = np.std(Baseline_scores_batch, axis = 1)

training_cat = list(['mixed', 'all grunts', 'all barks']) #['mixed', 'all_grunts', 'all_coos'] #['mixed', 'all grunts', 'all barks'] #['mixed', 'all_arch', 'all_coos']
#Get overall average

ov_avg_SFA = np.mean(avg_per_vocal_SFA,axis = 0)
ov_std_SFA = np.std(avg_per_vocal_SFA,axis = 0)

ov_avg_Baseline = np.mean(avg_per_vocal_Baseline,axis = 0)
ov_std_Baseline = np.std(avg_per_vocal_Baseline,axis = 0)

plt.figure()
plt.plot(np.array([1,2,3]),ov_avg_SFA,'b.')
plt.plot(np.array([1,2,3]),ov_avg_SFA[:] + ov_std_SFA[:],'b2')
plt.plot(np.array([1,2,3]),ov_avg_SFA[:] -1.0*ov_std_SFA[:],'b1')

plt.plot(np.array([1,2,3]),ov_avg_Baseline,'r.')
plt.plot(np.array([1,2,3]),ov_avg_Baseline[:] + ov_std_Baseline[:],'r2')
plt.plot(np.array([1,2,3]),ov_avg_Baseline[:] -1.0*ov_std_Baseline[:],'r1')
plt.axis([0, 4, .70, 1.01])
plt.xticks([ 1, 2,  3], training_cat)
plt.ylabel('Classification Accuracy')

#break down by testing vocals again.  Reset prefixes and other vars needed

prefixes = list(['gt', 'sb'])
call_counter1 = np.zeros([1,2]) #using matrix mult to get the combo identity of each pair
call_counter2 = np.zeros([1,2]) #doing it this way since I don't want to have to call newaxis each time
combo_counter = np.zeros([2,2]) 
call_ti_avg = np.zeros([3,2,2]) #add the performance for each pair to one spot in the matrix then get average by dividing by combo_counter
ctia_fold = np.zeros([3,2,2]) #this setup revealed no clear pattern.  Would like to get some sense of the variance but think will have to do that in the loop as things are different sizes

#get counts for each pair.

for pair in range(0,unpack_pairs.shape[1]):
        
        cur_pair_first = all_pairs[pair][0][0]
        cur_pair_second = all_pairs[pair][1][0]
        
        #create a vector which identifies which of the four calls that call is
        
        call_counter1[0,:] = np.array([cur_pair_first[0:2] == prefixes[0],cur_pair_first[0:2] == prefixes[1]])
        
        
        call_counter2[0,:] = np.array([cur_pair_second[0:2] == prefixes[0],cur_pair_second[0:2] == prefixes[1]])
            
                
        temp = np.matmul(call_counter1.T,call_counter2)
    
        combo_counter +=  temp  

#fold diagonal.  Trivial right now since 2x2 but leave general so we can change it later    
cc_fold = np.triu(combo_counter).T + np.tril(combo_counter) #add upper tri transposed to lower tri
np.fill_diagonal(cc_fold, np.diag(combo_counter)) #reset the main diagonal to be what it actually is.
cc_fold[cc_fold<1]=-1 #set zeros equal to -1 to prevent division by zero


#going to try something to check if the variance is different across the testing vocals
#going to try using a list of np. array and append values
#note will have to ultimately append the values in pos 1 and pos 2 (recall pos 0 is used) as both are coo vs grunt just in different orders.
#can check the number of elements vs the cc_fold above to be sure.

c_ti_hold = list([[],[],[]])
for ti in range(0,avg_per_vocal_SFA.shape[1]):
    c_ti_hold[ti] = list([np.array([]), np.array([]), np.array([]), np.array([])])

#before transition, flatten results of temp, use this to index c_ti_hold list and then append elements as needed.
for training_ind in range(0,avg_per_vocal_SFA.shape[1]): #finer grain analysis needed...see no difference amoungst training sets, but yet see pretty high variance across categories
    
    for pair in range(0,unpack_pairs.shape[1]):
        
        cur_pair_first = all_pairs[pair][0][0]
        cur_pair_second = all_pairs[pair][1][0]
        
        #create a vector which identifies which of the four calls that call is
        
        call_counter1[0,:] = np.array([cur_pair_first[0:2] == prefixes[0],cur_pair_first[0:2] == prefixes[1]])
        
        
        call_counter2[0,:] = np.array([cur_pair_second[0:2] == prefixes[0],cur_pair_second[0:2] == prefixes[1]])
        
        temp = np.matrix.flatten(np.matmul(call_counter1.T,call_counter2))
        
        call_ti_avg[training_ind,np.flatnonzero(call_counter1)[0],np.flatnonzero(call_counter2)[0]] += avg_per_vocal_SFA[pair,training_ind]
        
        c_ti_hold[training_ind][np.flatnonzero(temp)[0]] = np.append(c_ti_hold[training_ind][np.flatnonzero(temp)[0]],avg_per_vocal_SFA[pair,training_ind])
    #put in fold later, for now just look at variance across "4" categories   
    ctia_fold[training_ind,:,:] =  np.triu(call_ti_avg[training_ind,:,:]).T + np.tril(call_ti_avg[training_ind,:,:]) #add upper tri transposed to lower tri
    np.fill_diagonal(ctia_fold[training_ind,:,:], np.diag(call_ti_avg[training_ind,:,:])) #reset the main diagonal to be what it actually is.          
                
 
    plt.figure()
    plt.imshow(ctia_fold[training_ind,:,:]/cc_fold, extent=[0,1,0,1])
    plt.clim(0.5, 1)
    plt.colorbar()
    plt.xticks([ 0.25,   0.75], prefixes)
    plt.yticks([ 0.75,  0.25], prefixes) 
    plt.title('Accuracy for Training Group of ' + training_cat[training_ind])
    
    #put the coos vs grunts and grunts vs coos together.
    c_ti_hold[training_ind][1] = np.append(c_ti_hold[training_ind][1],c_ti_hold[training_ind][2]) 
    c_ti_hold[training_ind][2] = [] #just blank out the other one after appending
    #just make dummy variable for grouping for plots for each training ind
    together = list([[c_ti_hold[training_ind][0]], c_ti_hold[training_ind][1], [c_ti_hold[training_ind][3]]])
    #add to this in a sec but just take a look for now
    plt.figure()
    plt.boxplot(together)
    plt.xticks([1,2,3],['grunt v grunt', 'grunt v bark', 'bark v bark']) #['coo v coo', 'coo v grunt', 'grunt v grunt'] #['grunt v grunt', 'grunt v bark', 'bark v bark'] #['coo v coo', 'coo v arch', 'arch v arch']
    plt.ylabel('Classification Accuracy')
    plt.title(training_cat[training_ind])

#Show the call break down
plt.figure() 
plt.imshow(cc_fold, extent=[0,1,0,1])
plt.xticks([ 0.25,   0.75], prefixes)
plt.yticks([ 0.75,  0.25], prefixes)   
plt.title('Frequency of pairs')
plt.colorbar()
#Show average break down for each training condition