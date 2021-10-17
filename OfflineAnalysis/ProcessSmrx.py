import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
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
sys.path.append('C:/Users/bitsik0000/PycharmProjects/ClosedLoopEEG/OfflineAnalysis')
# from Mouse import *
from Config import *

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# -----------------------------------------------------------------




#Create directory to save figures
figureFolder = mh.gen_folder(EphysDir, Folder)
#Load EEG data
mh.read_smrx(FilePath)

# # evaluate quality of recordings
# matplotlib.use('Agg')
# plt.figure()
# plt.plot(mh.EEG_data)
# # plt.plot(fsig)
# # plt.plot(fsig2)
# plt.title('{}'.format(mh.Ch_name))
# plt.ylabel(mh.Ch_units)
# plt.ylim(1000,-1000)
# plt.savefig(figureFolder+'{}_{}'.format(mh.Ch_name,File[:6]) + mh.figure_tail)
# matplotlib.use('Qt5Agg')

### -------------------
### Downsample, perform multitaper and normalize data
target_fs=100
if mh.EEG_fs > target_fs:
    print ('downsampling mouse {} EEG data, from {}Hz to {}Hz'.format(mh.pos,mh.EEG_fs,target_fs))
    mh.downsample_EGG(target_fs=target_fs)


mh.multitaper(resolution=2)
mh.process_spectrum(smooth_iter=4,window_size=41)

