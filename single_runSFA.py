# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:46:13 2020

Code for doing single examples for looking at what features SFA finds or double checking
How things are running.

Need to put plotting functions back in...

I think issues with low SNR come down to there being gaps in the noise.  I.e. 
with no signal can end up with a training signal where one block has signal and one block has silence
which is still easy to separate.  Need to think on a way around this.

2020-07-21 way around above problem was changing clutter to have more vocs and 
eliminating any space in the clutter (i.e. there are no pauses of silence in clutter)
Changed downsampling so signals are more accurately represented.  Still see kind of odd results
going to redo checks with putting in the same vocalization twice and try white noise as well

May also try doing check with just noise to see what happens.


Things are still weird so trying to see what happens from very basic cases: Doing this to sb2933 and co2d14
    0)Putting in the same vocalization drops everything to chance as does putting in just noise
        -note this was 1 presentation, no gaps, no added noise.
    1) Vocals presented once, no gaps, no noise.  Order of vocalizations has largest effect on performance
    1.1) Vocals presented more than once, doesn't seem to really influence the performance
    2) Vocals presented once with gaps
        -gaps in training data interestingly improves performance, keeping with 1 presentation of vocals for now
    3) Vocals with gaps with testing noise only (i.e. should be bad)
        -classifer doesn't converge; moving on
    4)Vocals wih gaps and only training noise.
        -much more variance based on which vocalization comes first
        -this continues to hold true when increase training vocalizations to 2
        -surprisingly this seems more important than snr, so may want to just show vocalization once and then do both orders
        -trying to add test noise first though...seems to continue to be the case that order of training vocals matters the most
        -stike above, test noise creates odd behavior, turning off for now.
        
    5) repeating above with white noise.
-seems similarish, weirdest thing is as long as there are some vocalizations in with the nosie than there is some able to classify...going to switch from this to other work for the rest of the week and think about it more.
-started to run into issues with baseline classifer now though

-Next idea; try putting in two "different" white noise files as the vocalizations with no noise
--this results in chance performance!  Good sign.  Need to redo, put the same noise in twice by accident
...rate for both increase SFA goes to 72% and Baseline goes to 55%, this result replicates
 
-Now trying with vocal noise as clutter
--first training noise but no testing noise: drops to 0.536 0.542  SFA and 0.551 0.549 Baseline.  SNR 0.01 and 5
--now trainign noise and testing noise: oddly better with .689 and .645 for sfa and .85 and .67 for the baseline
--repeat get .631 .678 SFA 0.665 0.708 baseline
    
-Putting in same vocalization and noise as clutter still results in above chance performance.  Thus I think we should test sans noise in the future.
--i.e. I think the vocals as noise can drive the classifer in a way we dont want.
-Now trying to run sans testing noise but with different white noise as targets
--again right where we want to be.  Not a big impact for SNR changes, but at least we are seeing chance performance
---to test SNR a bit more added snr of 20 to whitenoise vs whitenoise1 set.
---SFA scores do go up with SNR, extend one more SNR and see if this repeats.
----It does and quite significantly so, good news!

-Next use white noise again as clutter and test on vocalizations sans testing noise
--Good news, now see high results, but trajectory to higher SNR is consistent and doesnt cap out as much.
--Going to continue by checking with messing with split size in split shuffle.
---Also going to formally add the plotting feature outside of the loop
---test 1, setting test_split to its default, 10%, looks like this is too easy, get 100%
---next trying 50-50 split, still too easy get 100% for all but first SNR
---now 75% split,looks like still too easy...getting consistent results, but things seem to high
---next trying 90% split, still a bit too high
--- so going for 95%...not as much of a gradient but still getting lower for .01
...will run twice **now making lowest SNR 0.001 on the last
...surprisingly not much gradient change for even lower snr going to change to 10^-5
...10e-5 and 10e-4 show lower performance.
---Now trying 99% split, though from above can probably get away with 95% split
...99 yields more of a gradient but get inconsistent results (highest SNR lower than second highest) and baseline classifer is not consistent
....baseline should be consistent when no test noise as the test set is the same accross all SNR
---trying several runs of 95% now.  Looks okay
---going to try 99 again but now with more splits, starting with 20.  If get consistent baseline will keep otherwise just use 95% and deal with lower gradient
...Great this seems to work and give a better gradient without increasing speed too much.  Trying again and see consistent result.
...Double bonus, increasing the n_splits seems much more defensible



-Next step is to try edge cases.  No noise is easy, but try training on noise then SFA classifier on vocals
--No noise for cur pair 76, get pretty consistent results of 97% for SFA and 91/92% for Baseline
--training on noise then testing on vocals still yields some classification, suggesting even on noise SFA can find some features...
---for follow up try trying on vocal clutter...interesting, in one of the cases had chance accuracy, in others had high 70's low 80's
...try running a few times with vocal clutter...see why above happened.  vocal clutter/noise gets scaled by SNR so first entry is really low vocals
...can probably use this to try even lower snr and get at silence idea mentioned below.
...run running with much lower SNR range

...will then run with 1 only for SRN to see same scale effect.  We do see this but it comes up at 1e-4
...then redo this with white noise, gradient picks up much faster


-cool now try on a couple other vocalizations just to check
...above pattern vis-a vis noise learning seems to hold over a few vocalizations, tried 76 16 and 65
...now try regular situation (i.e. with target vocals) and white noise
...seeing some disparity between 0.01 and 5 but then consistent.  Going to increase n_splits and see if this stablizes things.
....look okay trying to expand SNR range since not testing with novel noise yet
....This helped now seeing results we want, going to try with more vocalizations
....First good results were with vocal pair 65, trying 76 then 16
.....76 is good, slight ceiling effect but thats okay, next is 16 is okay


...Then repeat with cluter noise instead of white noise
....16 doesn't have nice low end
....76 looks pretty good
....65 looks great, just some ceiling
...Now trying reruns to see consistency
...16 3 times: lowest was .714, .715, .715 with this consistency probably don't need to run each vocal multiple times any more
...for sake of completion do 76 and 65
...76 3 times: lowest was 0.541, 0.560, 0.552 not as consistent so maybe will run 3 times
...65 3 times: lowest was 0.516, 0.509, .508, so 3 time seems to be fair but we are pretty accurate so ten seems overkill.  Got slight cieling here too some times

for tonight 2020-07-29, set up novel noise tests and then do visualization and abstract tomorrow.

-test with novel noise
..first pace with 65 looks promising, trying other first then do repeats
..16 looks okay, drops down to .66 for lowest
..76 looks weird now...have to test this more, but good to know it is a possibility.  Check for consistency another time.

note: if white noise provides consistent and more aligned results, may just stick with that for now...go over this with Yale.

Updates/update summary 2020-08-05
-moving forward with clutter batch which is currently running on kilosort
-added null (shuffled labels) to batch runs, need to see if it work
-initial visualization is done, can now finally do linear weights and quadratic weights
..no immediate pattern is obvious in the quadratic weights.
-will follow Yale's suggestion of putting in very simple and clear stimuli to make get an intial idea on weights are doing.
..on this note, will probably just try a combination of some sin waves or something else minimally complex to see what comes out.
...this will probably be implemented tomorrow.

2020-08-11 first run with new code looks good, null examples are running now on kilosort, testing out dummy examples.
right now doing most things as a toggle.  Before submitting paper should use submission as an excuse to clean code and make it more pythonic.
@author: ARL
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

#let's see if negative snr can be passed...doesn't seem like it
#may add 1e-7 to expand range/have low 
snr_values = np.array([1e-7, 1e-5, 1e-3, .1, 1.0, 10.0, 100.0, 1000.0]) # #standard range 
#clutter noise only snr 1e-7, 1e-6, 1e-5, 1e-4, 100,1000 snr range[ 1e-7, 1e-6, 1e-5, 1e-4, 100,1000]

#Set up folders for stimulus pairs.  Will have to change for kilosort, but going to try some local runs now.
#Path is designed for everything to be unix based file names.  Start with this then convert to windows at the end
if local < 1:
    
    vocal_foldername = Path("C:/Users/ARL/.spyder-py3/SFA_Manuscript2020/HauserVocalizations50kHzSampling/Wav")
    
    #Since we are loading this in once just set up as a windows path.
    vocpairfile =  PureWindowsPath("C:\\Users\\ARL\\.spyder-py3\\SFA_Manuscript2020\\21-Jul-2020_pairs_list.mat")
    
    #This will be used repeatedly so again leave as unix path and then switch to windows before loading
    clutterfoldername =Path("C:/Users/ARL/.spyder-py3/SFA_Manuscript2020/21-Jul-2020_clutter/")
    unpack_pairs = io.loadmat(vocpairfile)
    
else: #repeat of above but for development on my laptop
    vocal_foldername = PureWindowsPath("C:\\Users\\ronwd\\.spyder-py3\\SFA_PostCOSYNEAPP-master\\Monkey_Calls\\HauserVocalizations\\Monkey_Wav_50K\\Wav")
    
    clutterfoldername = PureWindowsPath("C:\\Users\\ronwd\\Desktop\\MATLAB\\PsychoPhysicsWithYale\\stimulus_generation_SFA_manu\\21-Jul-2020_clutter")
    unpack_pairs = io.loadmat('21-Jul-2020_pairs_list.mat') #some weird path error with local copy but doing this seems to work...nevermind just set the path wrong but leave it for now

#Template for loading in each clutter chorus.
clutterfiletemplate = '_ClutterChorus_20calls.wav'

all_pairs = unpack_pairs['listofpairs']

cur_pair = 65#just put in any vocalization
vocal_pair = all_pairs[cur_pair] 
print(vocal_pair)

#Save SFA classification accuracy and the baseline model score
SFA_save_score = np.zeros((1,snr_values.size)) 
Baseline_save_score = np.zeros((1,snr_values.size)) 
plt.close('all')

#special case toggles
mismatch_data = False; #Set to make training and test data entirely different
#in this case, make training data noise and test data vocalizations
toy_examples = True; #Instead of vocalizations load in the two toy examples

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
     
     if toy_examples: #if toy examples switch vocals to two different two exampels taken from original 2002 paper and Bellec et al 2016 paper
         #first try, just put in the same vocal twice to check what the weights do
         voc1 = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_PostCOSYNEAPP\\dummyexample1c.wav")
         voc2 = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_PostCOSYNEAPP\\dummyexample1c.wav")
        
     #set up list of vocal files to load
     vocal_files = [voc1, voc2]
     
     
     #Get the clutter file for this particular pair.  Add 1 to the cur_pair due to difference in indexing between matlab and python
     clutter_file = str(cur_pair+1)+clutterfiletemplate #'white_noise.wav' #
     clutter_file = Path(clutter_file)
    
     noise_file = clutterfoldername / clutter_file
     #Same as above now switch to windows path
     
     # ############ #for just putting noise in
     # #going to try putting in the same noise twice with the same vocal noise clutter file
     # clutter_file = 'white_noise1.wav' #second white noise file
     # clutter_file = Path(clutter_file)
     # noise_file1 = clutterfoldername / clutter_file
          
   
     # # #test just putting in noise the whole time
     # voc1 = PureWindowsPath(noise_file)
     # voc2 = PureWindowsPath(noise_file1)
     # vocal_files = [voc1, voc2]
     
     # #when putting just noise, try putting in vocalizations as clutter
     # clutter_file = str(cur_pair+1)+clutterfiletemplate
     # noise_file =  noise_file = clutterfoldername / clutter_file
     # noise_file = PureWindowsPath(noise_file)
     # #############
     

     
     num_vocals = len(vocal_files) #for use later, get number of unique stimulus files loaded

    ## Parameters for vocalization and noise pre processing
     signal_to_noise_ratio = snr_values[iteration]#scales by average power across noise and vocalizations
     #2020-07-28 :I think an artifical delay is being introduces by the filters.  At least the temporally filtered vocs don't look lined up...
     
     gfb = g.GammatoneFilterbank(order=1, density = 1.0, startband = -21, endband = 21, normfreq = 2200) #sets up parameters for our gammatone filter model of the cochlea.
                            #Need to look at documentation to figure out exactly how these parameters work , but normfreq at least seems to be central frequency from
                            #which the rest of the fitler a distributed (accoruding to startband and endband)
    # plot_gammatone_transformed = False #toggle to plot output of gammatone filtered stimulus
     #plot_temporal_filters = False #toggle to plot temporal filters (i.e. temporal component of STRF)
     #plot_temporal_transformed = True #toggle to plot signal after being gammatone filtered and temporally filtered 
     down_sample = True #down sample stimulus to help with RAM issues and general dimensionality issues.  I believe mostly reduces resolution of frequency
     #2020-07-20 trying to turn off downsampling (i.e. se this to 1, instead of 10), better if can run without doing this
     #This doesn't work on the local laptop but will try again on kilosort.
     #2020-07-21 now trying to work on seeing what is the lowest amount of downsampling
     #we can get away with.  For now leaving pre and post to match as this seems to influence performance?
     #I guess this introduces some kind of offset between the gamma filter and temporal filter ....need to figure this out at some point, but it is definitely the case
     
     #using a factor of 5 works
     #using a factor of 2 is possible but does cause the memory to peak out a bit on my laptop
     #sticking with factor of 2, going to check other vocalization but this seems to be lowest down sampling we can get away with
     
     
     #further good news, performance improves as downsampling is decreased which means the more accurate a rep of the signals we have the better we do
     down_sample_pre = 2 #Factor by which to reduce Fs by (e.g. 10 reduces Fs by one order of magnitude) 
     #2020-07-20 pre does the downsampling for the gamatone filter and post does down sampling for temporal filter
     down_sample_post = 2 #Factor by which to reduce Fs after applying filters 
        #(2019-09-18 trying removing either of the down sampling and it caused memory errors, meaning I don't fully understand how this is working)
        ## Parameters for training data
    
     
     ##Training and testing data parameters  
     num_samples = num_vocals * 1
     gaps = True #toggle whether there can be gaps between presentation of each stimulus
     apply_noise = True #toggle for applying noise to training data        
    #2020-07-28 if just do noise in testing, unable to converge (waited over 45 minutes)
     test_noise = False #Toggle for adding unique noise in test case that is different from training case
     
     if test_noise: #try loading in novel noise, in this case white noise
          clutter_file = 'white_noise1.wav' #second white noise file
          clutter_file = Path(clutter_file)
          noise_file1 = clutterfoldername / clutter_file
         
     
     #Maybe add in something to clear the classifiers in between iterations?  Do have general trend that later iterations become less accurate.  Perhaps above code doesn't reset classifier
     #just going to add this since it can't hurt
     #This did nothing.
     classifier_baseline = LinearSVC(max_iter = 10000000, tol = 0.001) #Perceptron(max_iter = 10000, tol = 0.001) #from scikit (presumably) #set up Perceptron classifier
     classifier_SFA = LinearSVC(max_iter = 10000000, tol = 0.001) #Perceptron(max_iter = 10000, tol = 0.001)
     
     
     
     #trying dropping classifier down to 5 featuers used 
     classifier_features = 5 #how many features from SFA  SFA classifer gets to use
     baseline_features = 'all' #how many features the Perceptron by itself gets to use
     
     ##plotting toggles
     plot_vocals = False #plot individual vocals after gamatone and temporal transformed
     plot_noise = False
     plot_training = False #plot training stream
     plot_test = False #plotting toggle for 
     plot_scores = True
     plot_features = True #plotting toggle for filters found by SFA
     plot_splits = False #plots split of data for the last iteration
     ## Load in files

     vocalizations = get_data(vocal_files) #get list object where each entry is a numpy array of each vocal file
     print('Vocalizations Loaded...')
    
    ##Load in and adjust noise power accordingly to signal to noise ratio
    
     if(load_noise):
        noise, _ = sf.read(noise_file)
        if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
            noise2, _ = sf.read(noise_file1)
    
     print('Noises loaded...')
     print('Ready for preprocessing.')
    
     if noise is not None:
        noise = scale_noise(vocalizations,noise,signal_to_noise_ratio) #scales based on average power
        noise = noise[:noiselen]
        if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
            noise2 = scale_noise(vocalizations,noise2, signal_to_noise_ratio)
     print('Noise Scaled...')
     print('Ready For Gammatone Transform')
    
    ## Apply Gammatone Transform to signal and noise
    
     vocals_transformed = gamma_transform_list(vocalizations, gfb) #does what is says on the tin: applies gamma transform to list of numpy arrays
     print('Vocalizations Transformed...')
    
     if noise is not None:
        noise_transformed = gamma_transform(noise, gfb)
        if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
            noise_transformed2 = gamma_transform(noise2,gfb)
        print('Noise Transformed...')
     if plot_vocals:
         for i in range(0,num_vocals):
             plt.figure()
             plt.imshow(vocals_transformed[i],aspect = 'auto', origin = 'lower')
             plt.title('Gammatone transformed')
    ## Down sample for computation tractablility
    
        
     if down_sample: #update 2020-01-21 downsample noise at this step too for our more structured noise
        for i,vocal in enumerate(vocals_transformed):
            vocals_transformed[i] = vocal[:,::down_sample_pre] #down samples by factor set in above code (e.g. 10 means reduce fs by one order of magnitude)
        if noise is not None:
            noise_transformed = noise_transformed[:,::down_sample_pre]
            if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
               noise_transformed2 = noise_transformed2[:,::down_sample_pre]
     print('Ready For Temporal Filters')
    
    ## Apply temporal filters
    #2020-07-21 double check that these filters are right and are not producing an abnormal offset between the narrower and longer filter
    #presumably these filters are reversed when convolve (like the normal case) so need to flip potentially when calculating the "STRF" for weights
     tFilter = temporalFilter()
     tFilter2 = np.repeat(tFilter,3)/3 #make wider filter
     tFilters = [tFilter, tFilter2]
     
     vocals_temporal_transformed = temporal_transform_list(vocals_transformed,tFilters)
     print('Vocals Temporally Filtered...')
     
     if noise is not None:
        noise_temporal_transformed = temporal_transform(noise_transformed,tFilters)
        print('Noise Temporally Filtered')
        if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
             noise_temporal_transformed2 = temporal_transform(noise_transformed2,tFilters)
             print('New Noise Temporally Filtered')
    #again re-evaluate if down sampled
        
     if down_sample:
        for i,vocal in enumerate(vocals_temporal_transformed):
            vocals_temporal_transformed[i] = vocal[:,::down_sample_post] #I guess this does a separate down sample after the temporal filters have been applied?
        if noise is not None:
            noise_temporal_transformed = noise_temporal_transformed[:,::down_sample_post]
            if test_noise: #could make noise code work with lists like other code, but just doing this quick change for now.
                noise_temporal_transformed2 = noise_temporal_transformed2[:,::down_sample_post]
     if plot_vocals:
         for i in range(0,num_vocals):
             plt.figure()
             plt.imshow(vocals_temporal_transformed[i],aspect = 'auto', origin = 'lower')
             plt.title('temporal transformed')
   
## Create Training Dataset
    #self note 2020-07-20, I want to change the above since it doesnt guarentee you see 5 of each vocalization
    #Update this won't exactly work for the invar stuff, but we can potentially use the original code for that
    
    #add toggle here for testing what happens when training data doesn't match testing data
    #in particular when train on just noise and test on vocals
     if mismatch_data: #If mismatch, replace vocals with noise file.
         
         training_data = noise_temporal_transformed
         
         if(gaps): # can keep the gaps the same
           min_gap = np.round(.05 * vocals_temporal_transformed[0].shape[1]) #sets min range of gap as percentage of length of a single vocalizations
           max_gap = np.round(.5 * vocals_temporal_transformed[0].shape[1]) #set max range of gap in same units as above
           training_data = np.concatenate((training_data, np.zeros((training_data.shape[0], np.random.randint(min_gap,max_gap)))),1)     
         print('Data arranged...')
         #Put in ability to apply noise, but will just shut off in noise case since it doesn't make a whole lot of sense.
         if(apply_noise): 
            while(noise_temporal_transformed[0].size < training_data[0].size):
                noise_temporal_transformed = np.hstack((noise_temporal_transformed,noise_temporal_transformed))
            if plot_noise:
                plt.figure()
                plt.title('Noise')
                plt.imshow(noise_temporal_transformed[:,0:training_data[0].size], aspect = 'auto', origin = 'lower')
                
            training_data = training_data + noise_temporal_transformed[:,0:training_data[0].size]
            print('Applied Noise...')
           
         else:
            print('No Noise Applied...')
         
     else: #If not mistmatch, run code normally
             
         samples = np.random.randint(num_vocals, size = num_samples)
         even_samples_check = np.sum(samples==1)
         #Again, note above comment,I think this only really works when only two vocalizations which is the case for now
         while even_samples_check != np.round(num_samples/num_vocals): #while samples are not even across vocalizations
               print('Ensuring equal presentation of both vocalizations')
               samples = np.random.randint(num_vocals, size = num_samples)
               even_samples_check = np.sum(samples==1)
         print('Equal presentation of both vocalizations established')
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
         #Don't really need this plot so commenting out.
         # if plot_training:
         #    plt.figure()
         #    plt.title('training_data')
         #    plt.imshow(training_data, aspect = 'auto', origin = 'lower')
    
         if(apply_noise): 
            while(noise_temporal_transformed[0].size < training_data[0].size):
                noise_temporal_transformed = np.hstack((noise_temporal_transformed,noise_temporal_transformed))
            if plot_noise:
                plt.figure()
                plt.title('Noise')
                plt.imshow(noise_temporal_transformed[:,0:training_data[0].size], aspect = 'auto', origin = 'lower')
                
            training_data = training_data + noise_temporal_transformed[:,0:training_data[0].size]
            print('Applied Noise...')
           
         else:
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
        #rolling noise doesn't make a big enough difference/want to try full novel noise
        
        # #roll noise by some random amount to make it not the same as training noise
        # min_shift = np.round(.05*testing_data[0].size) #shift at least 5%
        # max_shift = np.round(.5*testing_data[0].size) #at most shift 50%
        # new_noise = np.roll(noise_temporal_transformed[:,0:testing_data[0].size], np.random.randint(min_shift, max_shift))
        # testing_data = testing_data + new_noise[:,0:testing_data[0].size]
        
        testing_data = testing_data + noise_temporal_transformed2[:,0:testing_data[0].size]
        print('Applied Noise...')
     else:
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
    #test = np.vstack((test[:,5:], test[:,:-5]))
    #test = testSF(test, 'Layer 2', mean2, variance2, data_SS2, weights2)
    
    
    
     labels = getlabels(vocals_temporal_transformed)
    ## Get random inds to train the classifier and then random inds to test
    #Unforunately this won't quite work with the baseline classifier so just do this manually for now
    # class_train, class_test, labels_train, labels_test = model_selection.train_test_split(test, test_size = .1)
    #2020-07-16 set up cross validation loop manually, but this is looking better so far
    
     # all_inds = np.arange(0,np.shape(testing_data)[1])
     # test_inds = np.random.choice(all_inds,100,replace = False) #for now leave at 100 test points, can change to be a percentage later
     # train_inds = np.delete(all_inds,test_inds)
     
    ## Compare SFA With Baseline For Linear Classification
    #2020-07-20 note: maybe try stratified cross validation to ensure each cv gets the same number of points from both vocalizations?
    #Also may reduce cv to 5 due to varying size of stimuli
    #Trying reducing this down to 5
    
    #2020-07-28 Trying to use ShuffleSplit classifier since we have ordered vocals
    #See notes above which details notes that were originally here
    #found n_splits and 20 and test_size = .99 original the best now trying to expand n_splits to get more stability
     the_cv = ShuffleSplit(n_splits = 30, test_size = 0.99)
    
     print('SFA Based Classifier with ', classifier_features, ' features')
     #add a cv loop to pin down variance
     cv_sfa = cross_validate(classifier_SFA, test.T, labels,cv=the_cv)
     
     #classifier_SFA.fit(test[:classifier_features,train_inds].T,labels[train_inds])
     #print(classifier_SFA.score(test[:classifier_features].T,labels), '\n')
     #print(np.sum(labels[test_inds]==1)/np.size(test_inds)*100.0, ' percent voc 2 in testing data')
     print(cv_sfa['test_score'])
     print('Mean CV ', np.mean(cv_sfa['test_score']))
     
     SFA_save_score[0,iteration] = np.mean(cv_sfa['test_score'])
     #classifier_SFA.score(test[:classifier_features, test_inds].T,labels[test_inds]) #make this whole code into a for loop and save the scores for a particular SNR here
    
     print('Baseline Classifier with ', baseline_features, ' features')
     cv_baseline = cross_validate(classifier_baseline, testing_data.T, labels,cv=the_cv)
     #classifier_baseline.fit(testing_data[:,train_inds].T,labels[train_inds])
     #print(classifier_baseline.score(testing_data[:,test_inds].T,labels[test_inds]))
     print(cv_baseline['test_score'])
     print('Mean CV ', np.mean(cv_baseline['test_score']))
     
     Baseline_save_score[0,iteration] = np.mean(cv_baseline['test_score'])#classifier_baseline.score(testing_data.T,labels)
print('')
print('')    
print(SFA_save_score)

print('') 

print(Baseline_save_score)

if plot_scores:
    
    plt.figure()
    plt.plot(np.log10(snr_values), SFA_save_score[0,:]*100.0)
    plt.ylabel('CV Average Percent Classified Correct')
    plt.xlabel('log10 SNR')

if plot_splits: #note this spits out a lot of figures, there should be a work around but it is a bit hard since test_index is a pain to get out.
    #probably will have to switch this to saving these variable to another variable and then plotting a subset
    #for now just do one at a time
    for train_index, test_index in the_cv.split(test.T):
    
    
        plt.figure()
        plt.hist(labels[train_index])
        plt.title('training data distribution for each split')
    
    
    for train_index, test_index in the_cv.split(test.T):
    
    
        plt.figure()
        plt.hist(labels[test_index])
        plt.title('training data distribution for each split')

#Code chunk to look at weights.  Just do this for the last snr run for now and then edit code as you wish.
#See notes in getsf but essentially just need to do weights@data_ss, first testing_data.shape[0] terms are linear terms the rest are quad terms

if plot_features:
    
    #should probably put something in here to check figures then close other plots so don't get so many figures
    #for now just going to put a hardcoded close all
    #plt.close('all')
    
    for i in range(classifier_features): #quickly plot the used features over time on the same figure.  Simple move the figure generation to get the separately
        plt.figure()
        plt.plot(np.arange(0,test.shape[1]),test[i,:])
        #plt.plot() #Add functionality to plot vertical line where vocalizations change.
    
    features_to_plot = classifier_features #for now just use features used by classifer but can plot up to retain number of features (i.e. 20 in our case)
    
    weights_qs = weights @ data_SS #matmul weights by pcs to get weights back in quad expansion space
    #double check indices since been spending so much time with MATLAB
    
    weights_lin = weights_qs[0:features_to_plot,0:testing_data.shape[0]] #get all linear weights
    
    weights_quad = weights_qs[0:features_to_plot, testing_data.shape[0]::] #get all quadratic weights
    temp_quad = np.zeros([features_to_plot, weights_lin.shape[1],weights_lin.shape[1]])
    
    #Can't think of better idea so going to use some for loops for now.  Come back to trying to make this elegant later
     #first get ending index  for each features (note each column entry decreases by 1 in weight matrix as you go and not including diag of weight matrix yet)
     #Update 2020-08-05 this works
     
    end_idx = np.zeros([weights_lin.shape[1]-1])
    start_idx = np.zeros([weights_lin.shape[1]-1])
    counter = 0
    for i in range(0,weights_lin.shape[1]-1):
        counter = counter + 83 - i #get the reduced end_idx for each column of the triangle (i.e. corner) of the weight matrix (i.e. 83 elements then 82 elements...1 element)
        end_idx[i] = counter
        start_idx[i] = end_idx[i] - 83 + i
    
    for i in range(features_to_plot):
    
    
        for j in range(1,84): #fill in each column of corr matrix except diag
            #this check looks right now try it
           # print(weights_quad[i,int(start_idx[j-1]):int(end_idx[j-1])])
            
            temp_quad[i,j:temp_quad.shape[1],j-1] = weights_quad[i,int(start_idx[j-1]):int(end_idx[j-1])]
        np.fill_diagonal(temp_quad[i,:,:], 1)
    # #Show all linear weights together then add them to subplots below when got that working
    # plt.figure()
    # plt.imshow(weights_lin[:,:],aspect = 'auto')
    # plt.yticks(ticks = np.arange(0,features_to_plot), labels = np.arange(1,6))
    # plt.xlabel('frequency channel')
    # plt.ylabel('SFA feature')
    
    extent = [0, testing_data.shape[0], 0, 30]
    for i in range(features_to_plot):
        plt.figure() #Now think about how to show linear weights and quad weights on same figure
        plt.imshow(np.reshape(weights_lin[i,:],[1,testing_data.shape[0]]), extent = extent, interpolation = None, cmap = 'viridis')
        plt.yticks([])
        hold = temp_quad[i,:,:]
        plt.figure()
        plt.imshow(hold,interpolation = None, cmap = 'viridis')
        