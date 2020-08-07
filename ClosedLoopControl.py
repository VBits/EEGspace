''''
Copyright 2020, Vassilis Bitsikas, All rights reserved.
This file should be used for data analysis of perturbation experiments from a single mouse

Artifact handling
Artifacts are being detected based on z score in the 4 main frequencies independently.
The thresholds can be set manually for each frequency. The corresponding timepoints are then padded with NaN.
State space locations for these points are imputed by a rolling mean window during denoising before normalization.

Analysis
This script can be used to create three plots:
1. Circadian average count of SWS, Wake and REM
2. Circadian average of Delta and Sigma during sleep, Gamma and Theta during Wake and Theta during REM
3. Plot the perturbation trajectory on top of the baseline PCA
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
import datetime
import os
from pydpc import Cluster
sys.path.append('C:/Users/bitsik0000/PycharmProjects/delta_analysis/SleepAnalysisPaper')
from utils import *
import matplotlib as mpl
import matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
### -------------------
# -----------------------------------------------------------------
# Inputs

#provide genotype and rig position
mh = Mouse("TRAP",2)
#data directory
EphysDir = 'D:/Ongoing_analysis/'

#experiment directory and filename
Folder = '181008_TRAP_females_4/baseline/'
FileMat = '181008_000_baseline.mat'


# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 500
# -----------------------------------------------------------------
# Load data
mh.add_data(EphysDir + Folder,FileMat)

#Create directory to save figures
figure_tail =  ' - {} - {}.png'.format(mh.pos,mh.genotype)
date = datetime.datetime.now().strftime("%y%m%d")
figureFolder = EphysDir + Folder + 'Mouse_{}_{}/'.format(mh.pos,date)
if not os.path.exists(figureFolder):
    print ('Directory created')
    os.makedirs(os.path.dirname(figureFolder), exist_ok=True)
else:
    print ('Directory exists')
### -------------------
# evaluate quality of recordings
matplotlib.use('Agg')
plt.figure()
plt.plot(mh.EEG_data)
plt.title('EEG')
plt.ylabel('uV')
plt.savefig(figureFolder+'EEG' + figure_tail)
matplotlib.use('Qt5Agg')

### -------------------
# Perform stft and downsample
#2sec resolution
nperseg = 4*mh.EEG_fs
mh.sleep_bandpower(nperseg=nperseg, fs = mh.EEG_fs, EMG=False, LP_filter=True,iterations=1)

###################################################
#---------------------------------------
# PCA, this function will calculate state space distribution of all points
mh.PCA(normalizer =False,robust=True)

rand_idx = np.random.choice(len(mh.pC), size=40000,replace=False)

fig = plt.figure()
plt.scatter(*mh.pC.T[:,rand_idx], c='k', linewidths=0, alpha=0.4, s=4)
plt.title('PCA')
plt.savefig(figureFolder+'PCA using fast Gamma' + figure_tail, dpi=dpi)

#Plot current point  on top of state space and design decision boundaries