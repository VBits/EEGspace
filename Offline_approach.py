
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

# provide genotype and rig position
mh = Mouse("B6J",6)

#data directory
EphysDir = 'D:/Ongoing_analysis/'

Folder = '200702_B6J_BurrowingSD/'
FileMat = '200724_000_B6J_burrowingSD.mat'




# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 500
figure_tail = ' - {} - {}.png'.format(mh.pos, mh.genotype)
# -----------------------------------------------------------------
# Load data
mh.add_data(EphysDir + Folder,FileMat)
#Create directory to save figures
figureFolder = mh.gen_folder(EphysDir,Folder)
###------------------------
#OPTIONAL remove drifting baseline
#This might be making 60Hz noise worse, so it needs to be called on demand
_ , mh.EEG_data = iterative_savitzky_golay(mh,iterations=3)


### -------------------
### Downsample, perform multitaper and normalize data
# for i in [3,4,5,6,7]:
#     mh.pos = i
#     mh.add_data(EphysDir + Folder, FileMat)
if mh.EEG_fs > 100:
    print ('downsampling EEG data, mouse {}'.format(mh.pos))
    mh.downsample_EGG(target_fs=100)
mh.multitaper()
mh.process_spectrum()
