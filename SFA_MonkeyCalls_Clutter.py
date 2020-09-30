# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:50:18 2020

2020-01-20

Rewriting clutter (and invar in a way too) code for analyzing monkey calls  from Hausser

Ideas of things to test:
    
    1) See if it can do BSS with clutter made of a chorus of monkey calls
    2) Source or call invariance (i.e. invariance properties across different calls by the same monkey or different monkey's with the same call)
    
1) is probably the most viable but something with invariance would be nice as well/may have to repitch story depending on what we think of and the results
    
    
2020-01-21

I think second downsample post temporal filters is excessive and seems to really mess up the shape of the fields    
ALso first noise file has a gap in it.  Also I think Chethan's implementation of adding noise does not account for downsampling correctly.

Some weird behavior so thing still need to be investigated but so far looks like this could be viable

2020-01-23
Right now this is working unbelievably well with high accuracy for all but the lowest snr.  Potentially coos and grunts are just very easy to separate so trying different calls or more extensive noise

Also may set up shamma's A-B paradigm soon and try that as well...perhaps this weekend since tomorrow and Friday will be grid days

Seeing how this does with less features (i.e. 5 instead of 13)
-still did well overall...this is quite hard to believe but maybe this works this well.
Keep probing by reducing number of times the vocalizations are seen from 10 to 5
Reduced features to 4 and see more normal behavior relative to SNR, increasing rounds to 20 from 10 to see if this stabilizes some things

2020-01-27

Trying to test the clutter result again to be more confident it is a real result.

Also recall this does actually match what we saw before that 4 features was sufficient for this process.

1) coo vs harmonic arch: co1405 vs ha2unk with 4 features
    
    behavior is chaotic and doesn't clearly trend with the SNR, though this could just be in the number of features used
    
    Update: thinking we should maybe try having noise in test case as well and just use another noise file.
    also should have sanity check of doing white noise.
    
    Also results from invar may explain why even with chorus can learn something from that since the noise still actually has chatter in it rather than being circle shifted and all wonky like noise in artifical case
2020-01-09

    Trying circle shifted calls to make the din more noisy and similar to the artficial stimulus
    
    
2020-06-12 Thinking through setting up large batch simulations.  Designing 
what needs to be done now and then will probably create a new, easier interface when ready to
Starting with Clutter case since I think that make more sense.
    
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
from pycaret.classification import * 

snr_values = np.array([0.0001, 0.05, 0.5, 1.0, 5, 10, 15, 25, 50]) #snr range, run noiseless as well just separately since that is a toggle in the code

#snr_values = [0.0001 ,1 ,5 ,5 ,5 ,5 ,5]

SFA_save_score = np.zeros((20,snr_values.size)) #going to reset this up to do everything in one go.  Maybe loose some of details but then we can at least just leave this running and having all the scores saved in one spot
#the r is to add \\ to everything in the path string
vocal_foldername = PureWindowsPath("C:\\Users\\ronwd\\.spyder-py3\\SFA_PostCOSYNEAPP-master\\Monkey_Calls\\HauserVocalizations\\Monkey_Wav_50K\\Wav")
noise_foldername = PureWindowsPath("C:\\Users\\ronwd\\Desktop\\MATLAB\\PsychoPhysicsWithYale\\stimulus_generation_SFA_manu\\16-Jun-2020_clutter")

for a_round in range(0,20): #rounds are for averaging over for each set of snr (i.e. each snr is done 10 times)
    plt.close('all')
    print(a_round)
    
    plt.close('all')

    for iteration in range(0,snr_values.size):
    
        #quick sanity upgrade:
        #plt.close('all')
        #just copy these directly from matlab code.  Eventually will set up pipeline for this, but not needed now.
        #Now trying two grunts but by a different animal
        voc1 = '\\co1405.wav'
        voc2 = '\\ha2unk.wav'
        
        ## Files for vocalizations and noise
        load_noise = True; #toggle whether noises is generated or pulled in from a pre-generated file
        noiselen = 100000 #if loading in a pre-generated file, only take this many samples
        # noise = np.random.randn(noiselen) #old way of generating noise
        noise = True #toggle for whether testing with noise or not
        #['basic.wav', 'altered.wav']#
        vocal_files = [vocal_foldername + voc1, vocal_foldername + voc2] #set names of files to be played/trained and tested on
        noise_file = noise_foldername + '\\ClutterChorus_10calls_3.wav' #'bp_w_noise.wav' #file to load for noise
        num_vocals = len(vocal_files) #for use later, get number of unique stimulus files loaded
        
        ## Parameters for vocalization and noise pre processing
        signal_to_noise_ratio = snr_values[iteration]#scales by average power across noise and vocalizations
        print(signal_to_noise_ratio)
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
        if(plot_gammatone_transformed):
            for i,vocal in enumerate(vocals_transformed):
                plot_input(vocal, vocal_files[i])
            if noise is not None:
                plot_input(noise_transformed, 'Noise')
            
        print('Ready For Temporal Filters')
        
        ## Apply temporal filters
        #presumably these filters are reversed when convolve (like the normal case) so need to flip potentially when calculating the "STRF" for weights
        tFilter = temporalFilter()
        tFilter2 = np.repeat(tFilter,3)/3 #make wider filter
        tFilters = [tFilter, tFilter2]
        
        if(plot_temporal_filters):
            plt.figure()
            plt.plot(tFilter)
            plt.plot(tFilter2)
            plt.show()
        
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
                
        if(plot_temporal_transformed): #note: this plots the channeled responses (think like a spectrogram-ish) of the two temporal filters stacked on one another
            for i,vocal in enumerate(vocals_temporal_transformed):
                plot_input(vocal, vocal_files[i])
            
            if noise is not None:
                plot_input(noise_temporal_transformed, 'Noise')
        
        
        
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
#        plt.figure()
#        plt.imshow(training_data)
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
        
        #data = np.vstack((layer1[:,5:], layer1[:,:-5]))
        
        #(mean2, variance2, data_SS2, weights2) = getSF(data, 'Layer2')
        
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
        
        if(plot_test):
            plot_input(testing_data, 'Testing Data')
            
        print('Testing Data Ready')
        
        ## Apply SFA to Test Data, also toggles for using second layer
        
        test = testSF(testing_data, 'Layer 1', mean, variance, data_SS, weights)
        print('SFA Applied To Test Set')
        #test = np.vstack((test[:,5:], test[:,:-5]))
        #test = testSF(test, 'Layer 2', mean2, variance2, data_SS2, weights2)
        
        ## Plot SFA features
        
        labels = getlabels(vocals_temporal_transformed)
        if(plot_features):
            plt.figure() #added just to make sure this goes on its own figure
            for i in range(4):
                plt.plot(test[i])
                plt.plot(labels)
                plt.show() 
            print('SFA Features Plotted')
        else:
            print('Skipping Feature Plotting')
            
        ## Compare SFA With Baseline For Linear Classification
            
        print('SFA Based Classifier with ', classifier_features, ' features')
        classifier_SFA.fit(test[:classifier_features].T,labels)
        print(classifier_SFA.score(test[:classifier_features].T,labels), '\n')
        
        SFA_save_score[a_round,iteration] = classifier_SFA.score(test[:classifier_features].T,labels); #make this whole code into a for loop and save the scores for a particular SNR here
        
        print('Baseline Classifier with ', baseline_features, ' features')
        classifier_baseline.fit(testing_data.T,labels)
        print(classifier_baseline.score(testing_data.T,labels))