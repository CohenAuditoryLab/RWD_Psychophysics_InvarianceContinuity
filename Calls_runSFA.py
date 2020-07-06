# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:30:58 2020

This will need to be edited but for now going to have this be a module that is imported
And then call the run_SFA function to actually run SFA and return the classification accuracy scores

@author: ronwd
"""

import numpy as np
import matplotlib.pyplot as plt 
import soundfile as sf 
import pyfilterbank.gammatone as g
import scipy.ndimage.filters as filt
from sklearn import svm
from sklearn.linear_model import Perceptron
from tqdm import tqdm
import SFA_Tools.SFA_Sets as s
from SFA_Tools.SFA_Func import *
from pathlib import Path, PureWindowsPath

def run_SFA(cur_pair,vocal_foldername,vocal_pair,clutterfoldername, clutterfiletemplate):


    
    snr_values = np.array([0.001, 0.05, 0.5, 1.0, 5, 10, 15, 25, 50]) #snr range, run noiseless as well just separately since that is a toggle in the code
    
    #Save SFA classification accuracy and the baseline model score
    SFA_save_score = np.zeros((20,snr_values.size)) 
    Baseline_save_score = np.zeros((20,snr_values.size)) 
    
    for a_round in range(0,20): #rounds are for averaging over for each set of snr (i.e. each snr is done 10 times)
        print(a_round)
        
        for iteration in range(0,snr_values.size): #iterations for doing each snr
             print(snr_values[iteration])
             ## Files for vocalizations and noise
             load_noise = True; #toggle whether noises is generated or pulled in from a pre-generated file
             noiselen = 100000 #if loading in a pre-generated file, only take this many samples
             noise = True #toggle for whether testing with noise or not
             #this works using pathlib see outer file for explanation
             voc1 = vocal_foldername / vocal_pair[0][0]
             voc2 = vocal_foldername / vocal_pair[1][0]
             #convert to windows path i.e using \ instead of /
             voc1 = PureWindowsPath(voc1)
             voc2 = PureWindowsPath(voc2)
             #set up list of vocal files to load
             vocal_files = [voc1, voc2]
             #Get the clutter file for this particular pair.  Add 1 to the cur_pair due to difference in indexing between matlab and python
             clutter_file = str(cur_pair+1)+clutterfiletemplate
             clutter_file = Path(clutter_file)
             noise_file = clutterfoldername / clutter_file
             #Same as above now switch to windows path
             noise_file = PureWindowsPath(noise_file)
             
             num_vocals = len(vocal_files) #for use later, get number of unique stimulus files loaded
        
            ## Parameters for vocalization and noise pre processing
             signal_to_noise_ratio = snr_values[iteration]#scales by average power across noise and vocalizations
             
             gfb = g.GammatoneFilterbank(order=1, density = 1.0, startband = -21, endband = 21, normfreq = 2200) #sets up parameters for our gammatone filter model of the cochlea.
                                    #Need to look at documentation to figure out exactly how these parameters work , but normfreq at least seems to be central frequency from
                                    #which the rest of the fitler a distributed (accoruding to startband and endband)
             plot_gammatone_transformed = False #toggle to plot output of gammatone filtered stimulus
             plot_temporal_filters = False #toggle to plot temporal filters (i.e. temporal component of STRF)
             plot_temporal_transformed = False #toggle to plot signal after being gammatone filtered and temporally filtered 
             down_sample = True #down sample stimulus to help with RAM issues and general dimensionality issues.  I believe mostly reduces resolution of frequency
             down_sample_pre = 10 #Factor by which to reduce Fs by (e.g. 10 reduces Fs by one order of magnitude) 
             down_sample_post = 10 #Factor by which to reduce Fs after applying filters 
                #(2019-09-18 trying removing either of the down sampling and it caused memory errors, meaning I don't fully understand how this is working)
                ## Parameters for training data
             num_samples = num_vocals * 5 #note current results on screen were with 15 examples, trying one last go at three vocalization then going to cut loses for tonight and stop #choose how many times you see each stimulus
             gaps = True #toggle whether there can be gaps between presentation of each stimulus
             apply_noise = True #toggle for applying noise
                
                ## Parameters for testing data
             test_noise = False #Toggle for adding unique noise in test case that is different from training case
             plot_test = False #plotting toggle for 
             plot_features = False #plotting toggle for filters found by SFA
                
             classifier_baseline = Perceptron(max_iter = 10000, tol = 0.001) #from scikit (presumably) #set up Perceptron classifier
             classifier_SFA = Perceptron(max_iter = 10000, tol = 0.001)
             classifier_features = 10 #how many features from SFA  SFA-Perceptron gets to use
             baseline_features = 'all' #how many features the Perceptron by itself gets to use
             
             ## Load in files
        
             vocalizations = get_data(vocal_files) #get list object where each entry is a numpy array of each vocal file
             print('Vocalizations Loaded...')
            
            ##Load in and adjust noise power accordingly to sigal to noise ratio
            
             if(load_noise):
                noise, _ = sf.read(noise_file)
            
             print('Noises loaded...')
             print('Ready for preprocessing.')
            
             if noise is not None:
                noise = scale_noise(vocalizations,noise,signal_to_noise_ratio) #scales based on average power
                noise = noise[:noiselen]
             print('Noise Scaled...')
             print('Ready For Gammatone Transform')
            
            ## Apply Gammatone Transform to signal and noise
            
             vocals_transformed = gamma_transform_list(vocalizations, gfb) #does what is says on the tin: applies gamma transform to list of numpy arrays
             print('Vocalizations Transformed...')
            
             if noise is not None:
                noise_transformed = gamma_transform(noise, gfb)
                print('Noise Transformed...')
                
            ## Down sample for computation tractablility
            #reeval gammatone transform accordingly
                
             if down_sample: #update 2020-01-21 downsample noise at this step too for our more structured noise
                for i,vocal in enumerate(vocals_transformed):
                    vocals_transformed[i] = vocal[:,::down_sample_pre] #down samples by factor set in above code (e.g. 10 means reduce fs by one order of magnitude)
                if noise is not None:
                    noise_transformed = noise_transformed[:,::down_sample_pre]
            
             print('Ready For Temporal Filters')
            
            ## Apply temporal filters
            #presumably these filters are reversed when convolve (like the normal case) so need to flip potentially when calculating the "STRF" for weights
             tFilter = temporalFilter()
             tFilter2 = np.repeat(tFilter,3)/3 #make wider filter
             tFilters = [tFilter, tFilter2]
             
             vocals_temporal_transformed = temporal_transform_list(vocals_transformed,tFilters)
             print('Vocals Temporally Filtered...')
            
             if noise is not None:
                noise_temporal_transformed = temporal_transform(noise_transformed,tFilters)
                print('Noise Temporally Filtered')
            
            #again re-evaluate if down sampled
                
             if down_sample:
                for i,vocal in enumerate(vocals_temporal_transformed):
                    vocals_temporal_transformed[i] = vocal[:,::down_sample_post] #I guess this does a separate down sample after the temporal filters have been applied?
                if noise is not None:
                    noise_temporal_transformed = noise_temporal_transformed[:,::down_sample_post]
                    
## Create Training Dataset
        
             samples = np.random.randint(num_vocals, size = num_samples)
            
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
    
             if(apply_noise): 
                while(noise_temporal_transformed[0].size < training_data[0].size):
                    noise_temporal_transformed = np.hstack((noise_temporal_transformed,noise_temporal_transformed))
                training_data = training_data + noise_temporal_transformed[:,0:training_data[0].size]
                print('Applied Noise...')
    #            plt.figure()
    #            this_title = 'Training Stream with Noise SNR: ' +  str(signal_to_noise_ratio)
    #            plt.title(this_title)
    #            plt.imshow(training_data, aspect = 'auto', origin = 'lower')
             else:
                print('No Noise Applied...')
            
             print('Ready For SFA')
             
              ## Train SFA On Data
        
        
             (layer1, mean, variance, data_SS, weights) = getSF(training_data, 'Layer 1', transform = True)
             print('SFA Training Complete')
        
        
            ## Test Results
            
             samples = np.arange(num_vocals)
            
             testing_data = None
             initialized = False
             for i in tqdm(samples):
                if(not(initialized)):
                    testing_data = vocals_temporal_transformed[i]
                    initialized = True
                else:
                    testing_data = np.concatenate((testing_data, vocals_temporal_transformed[i]),1) 
             print('Data arranged...')
            
             if(test_noise):
                #roll noise by some random amount to make it not the same as training noise
                min_shift = np.round(.05*testing_data[0].size) #shift at least 5%
                max_shift = np.round(.5*testing_data[0].size) #at most shift 50%
                new_noise = np.roll(noise_temporal_transformed[:,0:testing_data[0].size], np.random.randint(min_shift, max_shift))
                testing_data = testing_data + noise_temporal_transformed[:,0:testing_data[0].size]
                print('Applied Noise...')
             else:
                print('No Noise Applied...')
            
                
             print('Testing Data Ready')
            
            ## Apply SFA to Test Data, also toggles for using second layer
            
             test = testSF(testing_data, 'Layer 1', mean, variance, data_SS, weights)
             print('SFA Applied To Test Set')
            #test = np.vstack((test[:,5:], test[:,:-5]))
            #test = testSF(test, 'Layer 2', mean2, variance2, data_SS2, weights2)
            
            ## Plot SFA features
            
             labels = getlabels(vocals_temporal_transformed)
          
            ## Compare SFA With Baseline For Linear Classification
                
             print('SFA Based Classifier with ', classifier_features, ' features')
             classifier_SFA.fit(test[:classifier_features].T,labels)
             print(classifier_SFA.score(test[:classifier_features].T,labels), '\n')
            
             SFA_save_score[a_round,iteration] = classifier_SFA.score(test[:classifier_features].T,labels) #make this whole code into a for loop and save the scores for a particular SNR here
            
             print('Baseline Classifier with ', baseline_features, ' features')
             classifier_baseline.fit(testing_data.T,labels)
             print(classifier_baseline.score(testing_data.T,labels))
             
             Baseline_save_score[a_round,iteration] = classifier_baseline.score(testing_data.T,labels)
        
        
        
                
             
    #not returning weights for now/will write another file or modify things when running other examples.
    return SFA_save_score, Baseline_save_score