# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:56:35 2020

Local plotting of results from kilosort

@author: ronwd
"""

import numpy as np
import matplotlib.pyplot as plt 
import SFA_Tools.SFA_Sets as s
from SFA_Tools.SFA_Func import *
from pathlib import Path, PureWindowsPath
import os
from scipy import stats, io


#Set file

results_file = PureWindowsPath("C:\\Users\\ronwd\\OneDrive\\Documents\\GitHub\\SFA_results_fromkilo\\try1_newcode_AUG.mat")

io.loadmat(results_file)



