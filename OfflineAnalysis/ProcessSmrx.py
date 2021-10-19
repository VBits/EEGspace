import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib

plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
import datetime
import os

sys.path.append('C:/Users/bitsik0000/PycharmProjects/ClosedLoopEEG/OfflineAnalysis')
# from Mouse import *
from Config import *

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# -----------------------------------------------------------------




#Create directory to save figures
figureFolder = m.gen_folder(EphysDir, Folder)
#Load EEG data
m.read_smrx(FilePath)

### -------------------
### Downsample, perform multitaper and normalize data
target_fs=100
if m.EEG_fs > target_fs:
    print ('downsampling mouse {} EEG data, from {}Hz to {}Hz'.format(m.pos,m.EEG_fs,target_fs))
    m.downsample_EGG(target_fs=target_fs)


m.multitaper(resolution=2)
m.process_spectrum(smooth_iter=4,window_size=41)

