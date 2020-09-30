# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 09:32:41 2019


Post-COSYNE application work on SFA.  Shaping this up to be a paper.

Action Items:

    -Think through and draft script calling architecture i.e., 
     figure out what calls what and the most efficient way to set this up so we can run through data easily
     
     Currently think overall script that deals with processing inputs and calls various other scripts for each figure in the paper
     
    -Set up and plan through figures that will be in the paper
        create separate file for generating all of these figures
        amass stimulus set for each of these figures
        
    -Develop better performance metric for comparison with basic linear classifier
        This applies particularly for the clutter case, but could have bearing on the invar case too depending on the classifier chosen
        Obviously have the simply performance metric of accuracy, but would really like to compare this accuracy to other classifiers in a rigorous or at least semi-rigorous way
        e.g. implement Eugenio's suggestion of a how many features needed for comparable performance to a different classifier
        May want to keep brainstorming though
        
        SFA distance metric for invariance seems good to me and not to hard to motivate, but be reflective about and prepare defense of critiques.
        It is just different enough for people to take exception, even though it arguably evokes exactly what people suggest when decoding neuronal populations
        (whether or not they mean a true population code or simply a collective code)
    
    Current plan:    
        
        "Two challenges the auditory system faces are BSS and Invariance..."
        "Generating a useful feature space = generating a feature space where these challenges are easily solved"
        
        Figure 1: Schematic and basic visual description of model
        (panels: spectrogram from gammatone, temporal filters,model output with classic stim like in Bellec et al.)
        
        "We apply this algorithm to this simple artifical stimuli to understand..."
        "First we explore BSS..."
        
        Figure 2: Clutter results with parabolic frequency sweeps
        (panels: base stimulus, clutter stimulus, (maybe base with clutter, but feels space wasting), figure from Cosyne Application (over SNR, over clutter amount))
        
        "Then we explore whether this feature space supports invariance"
        
        Figure 3: Invar results with parabolic frequency sweeps
        (panels: base stimulus, base stimulus with various transformations, figure from COSYNE application (expanded though to deal with more cases))
        note: For this figure will have to talk through whether we want to go all out with analyzed space (i.e. there are a lot of combinations of test and training set)
        
        "Confident that this works on simple artifical stimuli, we explored whether SFA would be successful with more naturalistic stimuli"
        
        Figure 4: Clutter results with monkey calls
        (panels: copy Figure 2)
        
        Figure 5: Invar results with monkey calls
        (panels: copy Figure 3)
        
        Figure 6: Speaker identification
        This would be a new one where it tries to do invariance with respect to acoustic event (i.e. determine source from multiple different calls from the space individual)
        It is unclear if this would work or works yet, so have to do a test run of this before officially adding it
    
    

@author: ARL
"""

