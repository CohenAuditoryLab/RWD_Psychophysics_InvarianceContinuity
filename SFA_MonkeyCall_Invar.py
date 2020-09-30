# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 13:39:23 2020

Setting up another chunk of code of trying the different ideas of invariance Yale has posed:
    
    -what does it do trained on the same vocalizations by the same monkey
    -differing traing to test with same vocalization or with new vocalizations

2020-01-26
Running some basic first pass tests now: ALL ARE WITHOUT NOISE
    1)Ability to generalized to types of vocalizations not seen before
        a. train two grunts, test new grunt and a coo
        -sees vocal 10 times in training, using 5 features, had average accuracy of 83.5% +-4%
        b. train two grunts, test coo and cop scream
         -sees vocal 10 times in training, using 5 features, had average accuracy of 76.8% +-2.6%%
         -upping to 10 features ups accuracy to 88.5% +- 2.3%
        c.train two grunts, test coo and harmonic arch
        -sees vocal 10 times in training, using 5 features, had average accuracy of 99.9% +-3.4%%
        d. reverse of a (note didn't save the other coo, cop scream, or arch so not a perfect reverse)
        -sees vocal 10 times in training, using 5 features, had average accuracy of 90.0%+-4.8%%
        -also note that because grunts are shorter and change more rapidly, absolute value of the SFA features is lower than it was for the most contiuous cases.
        e. reverse of b
        -sees vocal 10 times in training, using 5 features, had average accuracy of 75.0%+-6.4%%
        -upping to 10 features ups accuracy to 90.6% +- 3.0%
        f. reverse of c
        -sees vocal 10 times in training, using 5 features, had average accuracy of 61.2%+-12.2%%
        -kind of cool as this is the case where it hasn't seen that stimuli can evolve so fast and has been trained on the "slowest" stimuli so kind of cool that this is the failure case
        -upping to 10 features accuracy to 92.1% +- 1.7% 
        -seeing if poor results with 5 features generalizes to some other arches and coos
        --63.0% +- 4.1%, 74.7% +- 4.2%,79.8% +- 15.2%,75.1%+-6.1%
        
2020-01-27
Let's run some basic Shamma like stuff too just to see if this is working the way I think: In particular try some pure tone stuff just to make sure what I assume will happen will.
        
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

#Distance_Between =np.zeros((5,10))

stim_iter_max = 1; #how many stimuli sets are iterated over (to cover stimulus space)
repeat_iter_max = 10; # how many tiems a stimulus set is repeated (to see stability)

vocal_foldername = r'C:\Users\ronwd\.spyder-py3\SFA_PostCOSYNEAPP-master\Monkey_Calls\HauserVocalizations\Monkey_Wav_50K\Wav'
SFA_classifierperform = np.zeros([1,repeat_iter_max])

#list_of_names = ['Stimulus_Set_1/AM_Stimulus_3_2.wav','Stimulus_Set_1/AM_Stimulus_6_3.5.wav','Stimulus_Set_1/AM_Stimulus_10_5.5.wav',\
                 #'Stimulus_Set_1/AM_Stimulus_19_10.wav','Stimulus_Set_2/AM_Stimulus_1_1.wav'];
for a_stimulus in range(0,1):
    print(a_stimulus)
    for a_round in range(0,10):
        print(a_round)
        #quick sanity upgrade:
        plt.close('all')
        
        voctr1 =  '\\ha2480.wav'
        voctr2 =  '\\co1480.wav'
        
        vocte1 = '\\ag1c75.wav'
        vocte2 = '\\ag1e55.wav'
        
        ## Files for vocalizations and noise
        load_noise = False; #toggle whether noises is generated or pulled in from a pre-generated file
        noiselen = 100000 #if loading in a pre-generated file, only take this many samples
        # noise = np.random.randn(noiselen) #old way of generating noise
        noise = None #note this is not a true and false toggle but a none toggle #toggle for whether testing with noise or not
        #['basic.wav', 'altered.wav']#
        vocal_files_train = [vocal_foldername + voctr1, vocal_foldername +voctr2];
        vocal_files_test = [vocal_foldername + vocte2, vocal_foldername + vocte1];
        noise_file = 'bp_w_noise.wav' #file to load for noise
        num_vocals_train = len(vocal_files_train) #for use later, get number of unique stimulus files loaded
        num_vocals_test = len(vocal_files_test)
        
        ## Parameters for vocalization and noise preprocessing
        signal_to_noise_ratio = 50 #unclear if scales on moment by moment amplitude or by power (i.e. intergrated energy across frequencies)
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
        num_samples = num_vocals_train * 10 #choose how many times you see each stimulus
        gaps = True #toggle whether there can be gaps between presentation of each stimulus
        
        apply_noise = False #toggle for applying noise
        
        ## Parameters for testing data
        test_noise = False #unclear, I guess toggle for adding unique noise in test case that is different from training case?
        plot_test = True #plotting toggle for ?
        plot_features = True #plotting toggle for filters found by SFA
        
        classifier_baseline = Perceptron(max_iter = 10000, tol = 0.001) #from scikit (presumably) #set up Perceptron classifier
        classifier_SFA = Perceptron(max_iter = 10000, tol = 0.001)
        classifier_features = 5 #how many features from SFA  SFA-Perceptron gets to use
        baseline_features = 'all' #how many features the Perceptron by itself gets to use
        
        ## Load in files
        
        vocalizations_train = get_data(vocal_files_train) #get list object where each entry is a numpy array of each vocal file
        vocalizations_test = get_data(vocal_files_test)
        print('Vocalizations Loaded...')
        
        ##Load in and adjust noise power accordingly to sigal to noise ratio
        
        if(load_noise):
            noise, _ = sf.read(noise_file)
        
        print('Noises loaded...')
        print('Ready for preprocessing.')
        
        if noise is not None:
            noise = scale_noise(vocalizations_train,noise,signal_to_noise_ratio) #scales based on average power
            noise = noise[:noiselen]
        print('Noise Scaled...')
        print('Ready For Gammatone Transform')
        
        ## Apply Gammatone Transform to signal and noise
        
        vocals_transformed_train = gamma_transform_list(vocalizations_train, gfb) #does what is says on the tin: applies gamma transform to list of numpy arrays
        vocals_transformed_test = gamma_transform_list(vocalizations_test, gfb)
        print('Vocalizations Transformed...')
        
        if noise is not None:
            noise_transformed = gamma_transform(noise, gfb)
            print('Noise Transformed...')
            
        ## Down sample for computation tractablility
        #reeval gammatone transform accordingly
            
        if down_sample:
            for i,vocal in enumerate(vocals_transformed_train):
                vocals_transformed_train[i] = vocal[:,::down_sample_pre] #down samples by factor set in above code (e.g. 10 means reduce fs by one order of magnitude)
            for i,vocal in enumerate(vocals_transformed_test):
                vocals_transformed_test[i] = vocal[:,::down_sample_pre] #down samples by factor set in above code (e.g. 10 means reduce fs by one order of magnitude)    
        
        if(plot_gammatone_transformed): #add in ploting the test ones as well if you feel like it later
            for i,vocal in enumerate(vocals_transformed_train):
                plot_input(vocal, vocal_files_train[i])
            if noise is not None:
                plot_input(noise_transformed, 'Noise')
            
        print('Ready For Temporal Filters')
        
        ## Apply temporal filters
        
        tFilter = temporalFilter()
        tFilter2 = np.repeat(tFilter,3)/3 #slightly unlear what is going on here
        tFilters = [tFilter, tFilter2]
        
        if(plot_temporal_filters):
            plt.figure()
            plt.plot(tFilter)
            plt.plot(tFilter2)
            plt.show()
        
        vocals_temporal_transformed_train = temporal_transform_list(vocals_transformed_train,tFilters)
        vocals_temporal_transformed_test = temporal_transform_list(vocals_transformed_test,tFilters)
        print('Vocals Temporally Filtered...')
        
        if noise is not None:
            noise_temporal_transformed = temporal_transform(noise_transformed,tFilters)
            print('Noise Temporally Filtered')
        
        #again re-evaluate if down sampled
            
        if down_sample:
            for i,vocal in enumerate(vocals_temporal_transformed_train):
                vocals_temporal_transformed_train[i] = vocal[:,::down_sample_post] #I guess this does a separate down sample after the temporal filters have been applied?
            for i,vocal in enumerate(vocals_temporal_transformed_test):
                vocals_temporal_transformed_test[i] = vocal[:,::down_sample_post] #I guess this does a separate down sample after the temporal filters have been applied?    
            if noise is not None:
                noise_temporal_transformed = noise_temporal_transformed[:,::down_sample_post]
                
        if(plot_temporal_transformed): #same thing add a plotting option for test data if feel like it later
            for i,vocal in enumerate(vocals_temporal_transformed_train):
                plot_input(vocal, vocal_files_train[i])
            
            if noise is not None:
                plot_input(noise_temporal_transformed, 'Noise')
        
        print('Ready For SFA')
        
        ## Create Training Dataset
        
        samples = np.random.randint(num_vocals_train, size = num_samples)
        
        training_data = None
        initialized = False
        for i in tqdm(samples):
            if(not(initialized)):
                training_data = vocals_temporal_transformed_train[i]
                initialized = True
            else:
                training_data = np.concatenate((training_data, vocals_temporal_transformed_train[i]),1)
                
            if(gaps):
                min_gap = np.round(.05 * vocals_temporal_transformed_train[0].shape[1]) #sets min range of gap as percentage of length of a single vocalizations
                max_gap = np.round(.5 * vocals_temporal_transformed_train[0].shape[1]) #set max range of gap in same units as above
                training_data = np.concatenate((training_data, np.zeros((training_data.shape[0], np.random.randint(min_gap,max_gap)))),1)     
        print('Data arranged...')
        if(apply_noise): 
            while(noise_temporal_transformed[0].size < training_data[0].size):
                noise_temporal_transformed = np.hstack((noise_temporal_transformed,noise_temporal_transformed))
            training_data = training_data + noise_temporal_transformed[:,0:training_data[0].size]
            print('Applied Noise...')
        else:
            print('No Noise Applied...')
        
        print('Ready For SFA')
        
        ## Train SFA On Data, two layers in this example
        
        
        (layer1, mean, variance, data_SS, weights) = getSF(training_data, 'Layer 1', transform = True)
        print('SFA Training Complete')
        
        #data = np.vstack((layer1[:,5:], layer1[:,:-5]))
        
        #(mean2, variance2, data_SS2, weights2) = getSF(data, 'Layer2')
        
        ## Test Results
        
        samples = np.arange(num_vocals_test)
        #I think this part will have to be altered more thoroughly to fix the different inputs versus testing data
        testing_data = None
        initialized = False
        for i in tqdm(samples):
            if(not(initialized)):
                testing_data = vocals_temporal_transformed_test[i]
                initialized = True
            else:
                testing_data = np.concatenate((testing_data, vocals_temporal_transformed_test[i]),1) 
        print('Data arranged...')
        
        if(test_noise):
            testing_data = testing_data + noise_temporal_transformed[:,0:testing_data[0].size]
            print('Applied Noise...')
        else:
            print('No Noise Applied...')
        
        if(plot_test):
            plot_input(testing_data, 'Testing Data')
            plt.gca().invert_yaxis()
        print('Testing Data Ready')
        
        ## Apply SFA to Test Data, also toggles for using second layer
        
        test = testSF(testing_data, 'Layer 1', mean, variance, data_SS, weights)
        print('SFA Applied To Test Set')
        #test = np.vstack((test[:,5:], test[:,:-5]))
        #test = testSF(test, 'Layer 2', mean2, variance2, data_SS2, weights2)
        
        ## Plot SFA features
        
        labels = getlabels(vocals_temporal_transformed_test)
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
        
        print('Baseline Classifier with ', baseline_features, ' features')
        classifier_baseline.fit(testing_data.T,labels)
        print(classifier_baseline.score(testing_data.T,labels))
        
        SFA_classifierperform[0,a_round] = classifier_SFA.score(test[:classifier_features].T,labels)
        
        ##Plot it
        
        #SFAClassifiedPlot(test,classifier_SFA,labels[:-5])
        
        
        #need to fix to generalize to more stimuli, but for now this takes the values of the SFA algorithm for each feature used for the duration of the stimulus
        
#        SFA_Traj1 = test[0:classifier_features,0:vocals_temporal_transformed_test[0].shape[2]]
#        SFA_Traj2 = test[0:classifier_features,n[2] :]
#        
#        #also need to edit if stimuli are different durations and therefore have different lengths...
#        Distance_Between[a_stimulus,a_round] = np.average(np.sqrt(np.sum((SFA_Traj1 - SFA_Traj2)**2, axis = 0)))