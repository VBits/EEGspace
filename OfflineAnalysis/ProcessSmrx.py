import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
import pandas as pd
import joblib
import datetime
import os
import sys
sys.path.append('C:/Users/bitsik0000/PycharmProjects/ClosedLoopEEG/OfflineAnalysis')
# from Mouse import *
from OfflineAnalysis.Config import *
from OfflineAnalysis.Mouse import Mouse

def get_mouse(strain,pos,load=True):
    m = Mouse(strain, pos)

    # Create directory to save figures
    m.figureFolder = m.gen_folder(EphysDir, Folder)
    if load:
        print('Loading previously analysed file _{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], m.genotype, m.pos))
        # Load previously saved Dataframe from experimental folder
        m.Sxx_df = pd.read_pickle(EphysDir + Folder + 'Sxx_df_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], m.genotype, m.pos))
        m.multitaper_df = pd.read_pickle(EphysDir + Folder + 'Multitaper_df_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], m.genotype, m.pos))
    else:
        print('Processing EEG data and storing files: _{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], m.genotype, m.pos))
        #Load EEG data
        m.read_smrx(FilePath)

        ### -------------------
        ### Downsample, perform multitaper and normalize data
        target_fs=100
        if m.EEG_fs > target_fs:
            print ('downsampling mouse {} EEG data, from {}Hz to {}Hz'.format(m.pos,m.EEG_fs,target_fs))
            m.downsample_EGG(target_fs=target_fs)


        m.multitaper(resolution=2)
        m.process_spectrum(window_size=21)

        # Save normalized Dataframe to experimental folder
        m.Sxx_df.to_pickle(EphysDir + Folder + 'Sxx_df_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], m.genotype, m.pos))
        m.multitaper_df.to_pickle(EphysDir + Folder + 'Multitaper_df_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], m.genotype, m.pos))

    return m