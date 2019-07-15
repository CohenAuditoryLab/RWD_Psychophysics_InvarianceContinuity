# -*- coding: utf-8 -*-
"""
Spyder Editor

2019-06-24

This is a temporary script file.

Temp copy from github just to see how it works/check that my computer has all the necessary parts
"""


import numpy as np
import matplotlib.pyplot as plt 
import soundfile as sf 
import scipy.ndimage.filters as filt
from sklearn import svm
from sklearn.linear_model import Perceptron
from tqdm import tqdm

import SFA_Tools.SFA_Sets as s
from SFA_Tools.SFA_Func import *
import pyfilterbank.gammatone as g

"################################################################################"
"################################################################################"
"################################################################################"

"Initialize Variables"

## Files for vocalizations and noise
load_noise = True
noise = None
noiselen = 100000
noise_file_folder = "C:\\Users\\ronwd\\.spyder-py3\\SFA_Local_Test_2019-06\\SFA\\"
file_folder = "C:\\Users\\ronwd\\Desktop\\MATLAB\\PsychoPhysicsWithYale\\Run_task_RWD_v3\\" 
vocal_files = ['test1.WAV','test2.WAV','test3.WAV','test4.WAV']#['coo.WAV','grunt.WAV','AG493B.WAV','CS1E54.WAV']
noise_file = 'Matlab_SoundTextureSynth\\Output_Folder\\Bubbling_water_10111010100.wav'
num_vocals = len(vocal_files)

## Parameters for vocalization and noise preprocessing
signal_to_noise_ratio = 50
gfb = g.GammatoneFilterbank(order=1, density = 1.0, startband = -21, endband = 21, normfreq = 2200)
plot_gammatone_transformed = False
plot_temporal_filters = False
plot_temporal_transformed = False

## Parameters for training data
num_samples = num_vocals * 30
gaps = True
min_gap = 25
max_gap = 100
apply_noise = False
skip = 40

## Parameters for testing data
test_noise = False
plot_test = True
plot_features = True

classifier_baseline = Perceptron(max_iter = 1000, tol = 0.001)
classifier_SFA = Perceptron(max_iter = 1000, tol = 0.001)
classifier_features = 2

for (i, file) in enumerate(vocal_files):
    vocal_files[i] = file_folder + vocal_files[i]
noise_file = noise_file_folder + noise_file

"################################################################################"
"################################################################################"
"################################################################################"

"Load In Files Containing Vocalizations And Noise"

vocalizations = get_data(vocal_files)
print('Vocalizations Loaded...')

if(load_noise):
    noise, _ = sf.read(noise_file)

print('Noises loaded...')
print('Ready for preprocessing.')

"################################################################################"
"################################################################################"
"################################################################################"

"Preprocess Vocalizations And Noise"

noise = scale_noise(vocalizations,noise,signal_to_noise_ratio)
noise = noise[:noiselen]
print('Noise Scaled...')
print('Ready For Gammatone Transform')

"################################################################################"
"################################################################################"
"################################################################################"

'2019-06-24 suddenly gfb is not defined in between these two lines.'
'Going with dumb fix of just reinitalizing gfb (also note that gfb object does'
'not appear in spyder variable explorer since it is not supported)'
'update the smple fix did not work going to need to figure out something else'


vocals_transformed = gamma_transform_list(vocalizations, gfb)
print('Vocalizations Transformed...')

noise_transformed = gamma_transform(noise, gfb)
print('Noise Transformed...')

if(plot_gammatone_transformed):
    for i,vocal in enumerate(vocals_transformed):
        plot_input(vocal, vocal_files[i])
    plot_input(noise_transformed, 'Noise')
    
print('Ready For Temporal Filters')

"################################################################################"
"################################################################################"
"################################################################################"

tFilter = temporalFilter()
tFilter2 = np.repeat(tFilter,3)/3
tFilters = [tFilter, tFilter2]

if(plot_temporal_filters):
    plt.plot(tFilter)
    plt.plot(tFilter2)
    plt.show()

vocals_temporal_transformed = temporal_transform_list(vocals_transformed,tFilters)
for (i,vocal) in enumerate(vocals_temporal_transformed):
    vocals_temporal_transformed[i] = vocal[:,::skip]
print('Vocals Temporally Filtered...')

noise_temporal_transformed = temporal_transform(noise_transformed,tFilters)
noise_temporal_transformed = noise_temporal_transformed[:,::skip]
print('Noise Temporraly Filtered')

if(plot_temporal_transformed):
    for i,vocal in enumerate(vocals_temporal_transformed):
        plot_input(vocal, vocal_files[i])
    plot_input(noise_temporal_transformed, 'Noise')

print('Ready For SFA')

"################################################################################"
"################################################################################"
"################################################################################"

"Creating Training Dataset"

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

"################################################################################"
"################################################################################"
"################################################################################"

"Train SFA on Data"

(mean, variance, data_SS, weights) = getSF(training_data, 'Layer 1')

print('SFA Training Complete')

"################################################################################"
"################################################################################"
"################################################################################"

"Test SFA Results"

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

"################################################################################"
"################################################################################"
"################################################################################"

if(test_noise):
    testing_data = testing_data + noise_temporal_transformed[:,0:testing_data[0].size]
    print('Applied Noise...')
else:
    print('No Noise Applied...')
    
if(plot_test):
    plot_input(testing_data, 'Testing Data')
print('Testing Data Ready')

test = testSF(testing_data, 'Layer 1', mean, variance, data_SS, weights)
print('SFA Applied To Test Set')

"################################################################################"
"################################################################################"
"################################################################################"

"Plot SFA Features"

labels = getlabels(vocals_temporal_transformed)

if(plot_features):
    for i in range(4):
        plt.figure(i)
        plt.plot(test[i])
        plt.plot(labels)
        plt.show() 
    print('SFA Features Plotted')
else:
    print('Skipping Feature Plotting')

"################################################################################"
"################################################################################"
"################################################################################"

"Compare SFA With Baseline For Linear Classification"
print('SFA Based Classifier with ', classifier_features, ' features')
classifier_SFA.fit(test[:classifier_features].T,labels)
print(classifier_SFA.score(test[:classifier_features].T,labels), '\n')

print('Baseline Classifier')
classifier_baseline.fit(testing_data.T,labels)
print(classifier_baseline.score(testing_data.T,labels))


SFAClassifiedPlot(test,classifier_SFA,labels)