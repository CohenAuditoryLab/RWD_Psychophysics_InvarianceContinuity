# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:08:04 2020

Invariance rewrite of single_runSFA

Combining these may be a decent idea but for now just having a separate, self contained file for each seems to keep things cleaner

For now going to leave out parts dealing with noise.  Can come back and just drop them in later if needed

Getting this to work to look at some results then run the major one when kilosort is back online

@author: ronwd
"""

import numpy as np
import matplotlib.pyplot as plt 
import soundfile as sf 
import pyfilterbank.gammatone as g
import scipy.ndimage.filters as filt
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn import model_selection
from tqdm import tqdm
import SFA_Tools.SFA_Sets as s
from SFA_Tools.SFA_Func import *
from pathlib import Path, PureWindowsPath
from scipy import stats, io
from sklearn.model_selection import cross_validate, ShuffleSplit

local = 1 #toggle between local code development on laptop and running code on Kilosort

if local < 1:
    
    
    vocal_foldername = Path("C:/Users/ARL/.spyder-py3/SFA_Manuscript2020/HauserVocalizations50kHzSampling/Wav")
    print("need to set new paths for non local code") 
    
    
else: #repeat of above but for development on my laptop

    vocal_foldername = PureWindowsPath("C:\\Users\\ronwd\\.spyder-py3\\SFA_PostCOSYNEAPP-master\\Monkey_Calls\\HauserVocalizations\\Monkey_Wav_50K\\Wav")
    
    #Since we are loading this in once just set up as a windows path.
    vocpairfile =  PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_PostCOSYNEAPP\\Invar_Stim_juneon\\18-Aug-2020_pairs_list.mat")
    
    trainingvocfile = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_PostCOSYNEAPP\\Invar_Stim_juneon\\18-Aug-2020_traininggroups.mat")

    unpack_pairs = io.loadmat(vocpairfile)
    unpack_training = io.loadmat(trainingvocfile)
   
   
#Template for loading in each clutter chorus.

all_pairs = unpack_pairs['listofpairs']
all_training = unpack_training['Training_Vocals']

cur_pair = 22#just put in any vocalization
vocal_pair = all_pairs[cur_pair] 
training_set = all_training[cur_pair] #index is weird for this due to conversion from matlab cell array to numpy array
#there may be a more elegant way to flatten this out, but for now, it goes [pair][set][vocal][0][0] with the last two needed to get the string out

print(vocal_pair)
#Save SFA classification accuracy and the baseline model score
SFA_save_score = np.zeros((1,training_set.size)) 
Baseline_save_score = np.zeros((1,training_set.size)) 
plt.close('all')

#this works using pathlib see outer file for explanation
voc1 = vocal_foldername / vocal_pair[0][0]
voc2 = vocal_foldername / vocal_pair[1][0]
#convert to windows path i.e using \ instead of /
voc1 = PureWindowsPath(voc1)
voc2 = PureWindowsPath(voc2)

#set up list of vocal files to load for testing data
vocal_files = [voc1, voc2]



#need to put for loop starting here to loop through training sets

for training_group in range(0,training_set.size): #for each training set


    #Unfortunately don't really have an elegent solution for setting up all of the training files we want to pull...so trying below
    training_files = list()
    for i in range(0, training_set[0].size): #grab all vocal files in that training set (10 in our case as of 2020-08-18)
    #Note training set is now [set][vocal][0][0]
    #2020-08-25 below works
        t_voc = vocal_foldername / training_set[training_group][i][0][0]
        t_voc = PureWindowsPath(t_voc)
        
        training_files.append((t_voc))
        
    'from here on we have to be careful to make sure things match up properly'
    
    #set number of vocals to the training vocals since this is called throughout code.  May have to have another variable for testing vocals though
    num_vocals = len(training_files) 
    num_test_vocals = len(vocal_files)
    
    #set up gammatone filter
    gfb = g.GammatoneFilterbank(order=1, density = 1.0, startband = -21, endband = 21, normfreq = 2200)
    
    down_sample = True #down sample stimulus to help with RAM issues and general dimensionality issues.  I believe mostly reduces resolution of frequency
 
 
 
    down_sample_pre = 2 #Factor by which to reduce Fs by (e.g. 10 reduces Fs by one order of magnitude) 
 
    down_sample_post = 2 #Factor by which to reduce Fs after applying filters 
    

 
     ##Training and testing data parameters  
    num_samples = num_vocals * 1
    gaps = True #toggle whether there can be gaps between presentation of each stimulus
    
    #skipping noise parameters for now
    #Set up classifiers
    classifier_baseline = LinearSVC(max_iter = 10000000, tol = 0.001) #Perceptron(max_iter = 10000, tol = 0.001) #from scikit (presumably) #set up Perceptron classifier
    classifier_SFA = LinearSVC(max_iter = 10000000, tol = 0.001) #Perceptron(max_iter = 10000, tol = 0.001)
    
    #trying dropping classifier down to 5 featuers used 
    classifier_features = 5 #how many features from SFA  SFA classifer gets to use
    baseline_features = 'all' #how many features the Perceptron by itself gets to use
 
    ##plotting toggles
    plot_vocals = False #plot individual vocals after gamatone and temporal transformed
    plot_training = False #plot training stream
    plot_test = False #plotting toggle for 
    plot_scores = True
    plot_features = False #plotting toggle for filters found by SFA
    plot_splits = False #plots split of data for the last iteration
    ## Load in files
     
    ## Load in files

    vocalizations = get_data(training_files) #get list object where each entry is a numpy array of each vocal file
    testvocalizations = get_data(vocal_files)
    print('Vocalizations Loaded...')
    
    ## Apply Gammatone Transform to training and test
    
    vocals_transformed = gamma_transform_list(vocalizations, gfb) #does what is says on the tin: applies gamma transform to list of numpy arrays
    
    testvocals_transformed = gamma_transform_list(testvocalizations, gfb) 
    print('Vocalizations Transformed...')
    
    if plot_vocals:
         for i in range(0,num_vocals):
             plt.figure()
             plt.imshow(vocals_transformed[i],aspect = 'auto', origin = 'lower')
             plt.title('Gammatone transformed')
    
     ## Down sample for computation tractablility
    
     
    if down_sample:
          for i,vocal in enumerate(vocals_transformed):
              vocals_transformed[i] = vocal[:,::down_sample_pre]
          for i, vocal in enumerate(testvocals_transformed):
              testvocals_transformed[i] = vocal[:,::down_sample_pre]
              
    print('Ready For Temporal Filters')
     
      ## Apply temporal filters
      #2020-07-21 double check that these filters are right and are not producing an abnormal offset between the narrower and longer filter
      #presumably these filters are reversed when convolve (like the normal case) so need to flip potentially when calculating the "STRF" for weights
    tFilter = temporalFilter()
    tFilter2 = np.repeat(tFilter,3)/3 #make wider filter
    tFilters = [tFilter, tFilter2]
     
    vocals_temporal_transformed = temporal_transform_list(vocals_transformed,tFilters)
    testvocals_temporal_transformed = temporal_transform_list(testvocals_transformed,tFilters)
    print('Vocals Temporally Filtered...')
     
    if down_sample:
        for i,vocal in enumerate(vocals_temporal_transformed):
            vocals_temporal_transformed[i] = vocal[:,::down_sample_post] 
        for i,vocal in enumerate(testvocals_temporal_transformed):
            testvocals_temporal_transformed[i] = vocal[:,::down_sample_post]
        
    if plot_vocals:
         for i in range(0,num_vocals):
             plt.figure()
             plt.imshow(vocals_temporal_transformed[i],aspect = 'auto', origin = 'lower')
             plt.title('temporal transformed')
             
    samples = np.random.choice(num_vocals, num_samples, replace=False) #Have to switch to using random.choice so can remove replacement.  This means we can remove while loop too
    print('Equal presentation of vocalizations established') #note this mainly works because we are presenting each vocal once.  If that was not the case we would have to set up some kind of loop or additional code
    
    training_data = None
    initialized = False
    for i in tqdm(samples):
        if(not(initialized)):
                   training_data = vocals_temporal_transformed[i]
                   initialized = True
        else:
                   training_data = np.concatenate((training_data, vocals_temporal_transformed[i]),1)
                   
        if(gaps):
                   min_gap = np.round(.05 * vocals_temporal_transformed[0].shape[1]) #sets min range of gap as percentage of length of a single vocalizations
                   max_gap = np.round(.5 * vocals_temporal_transformed[0].shape[1]) #set max range of gap in same units as above
                   training_data = np.concatenate((training_data, np.zeros((training_data.shape[0], np.random.randint(min_gap,max_gap)))),1)     
    print('Data arranged...')
    
    print('No Noise Applied...')    
    
    print('Ready For SFA')
    if plot_training:
            plt.figure()
            this_title = 'Training Stream with Noise SNR: ' +  str(signal_to_noise_ratio)
            plt.title(this_title)
            plt.imshow(training_data, aspect = 'auto', origin = 'lower')
            
            
    ## Train SFA On Data        
    (layer1, mean, variance, data_SS, weights) = getSF(training_data, 'Layer 1', transform = True)
    print('SFA Training Complete')
    
    ## Test Results 
    'NOTE: 220-08-25 This is the section we need to edit and check since training and test do not match'
    samples = np.arange(num_test_vocals)
    
    testing_data = None
    initialized = False
    for i in tqdm(samples):
        if(not(initialized)):
            testing_data = testvocals_temporal_transformed[i]
            initialized = True
        else:
            testing_data = np.concatenate((testing_data, testvocals_temporal_transformed[i]),1) 
    print('Data arranged...')
     
    print('No Noise Applied...')
    
     
    print('Testing Data Ready')
    if plot_test:
         plt.figure()
         this_title = 'Testing Stream with Noise SNR: ' +  str(signal_to_noise_ratio)
         plt.title(this_title)
         plt.imshow(testing_data, aspect = 'auto', origin = 'lower')
    
    ## Apply SFA to Test Data, also toggles for using second layer
    
    test = testSF(testing_data, 'Layer 1', mean, variance, data_SS, weights)
    print('SFA Applied To Test Set')
    
    labels = getlabels(testvocals_temporal_transformed)
    
    the_cv = ShuffleSplit(n_splits = 30, test_size = 0.99)
    
    print('SFA Based Classifier with ', classifier_features, ' features')
     #add a cv loop to pin down variance
    cv_sfa = cross_validate(classifier_SFA, test.T, labels,cv=the_cv)
     
  
    print(cv_sfa['test_score'])
    print('Mean CV ', np.mean(cv_sfa['test_score']))
     
    SFA_save_score[0,training_group] = np.mean(cv_sfa['test_score'])
     
    
    print('Baseline Classifier with ', baseline_features, ' features')
    cv_baseline = cross_validate(classifier_baseline, testing_data.T, labels,cv=the_cv)

    print(cv_baseline['test_score'])
    print('Mean CV ', np.mean(cv_baseline['test_score']))
     
    Baseline_save_score[0,training_group] = np.mean(cv_baseline['test_score'])#classifier_baseline.score(testing_data.T,labels)
     
print('')
print('')    
print(SFA_save_score)

print('') 

print(Baseline_save_score)
      

#Need to figure out something else for plotting, probably just do bar graphs and have something to add what the test vocals are