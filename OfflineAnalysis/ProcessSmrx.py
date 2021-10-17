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

#TODO provide genotype and rig position
# mh = Mouse("TRAP",2)
# mh = Mouse("MCH",3)
# mh = Mouse("B6J",8)
# for n in [9,10,11,12,13,14,15,16]:
for n in [2,3,4,5,6]:
    print (n)
    # mh = Mouse("Smad1PVcre",12)
    # mh = Mouse("B6Jv", 9)
    mh = Mouse("B6J", 1)
    # mh = Mouse('Nms',12)
    # mh = Mouse('Galcre',n)
    # mh = Mouse('Vglut2Cre-SuM', n)
    # mh = Mouse('SertCre-CS', n)
    mh = Mouse("Smad1fl-PVcre", 13)
    #data directory
    EphysDir = 'D:/Project_Mouse/Ongoing_analysis/'

    # # # #experiment directory and filename
    # # # Folder = '181008_TRAP_females_4/baseline/'
    # # # FileMat = '181008_000_baseline.mat'
    # # # Folder = '181008_TRAP_females_4/experiment/'
    # # # FileMat = '181017_000_experiment.mat'
    # # # #experiment directory and filename
    # # # Folder = '191028_MCHCre_hM4Di_antistatic/'
    # # # FileMat = '191122_002.mat'
    # # # #
    # # # Folder = '180817_B6J_NovelObjects/'
    # # # FileMat = '180817_001_mat.mat'
    # # # Folder = '170511_13_SD_water/'
    # # # FileMat = '170511_13_SD_water.mat'
    # # Folder = '200604_B6J_NO_Misting/'
    # # FileMat = '200604_2_NOandMisting.mat'
    # # # FileMat = '200604_1_NOandMisting.mat'
    # # Folder = '200428_Nms_DD/'
    # # # FileMat = '200428_baseline1.mat'
    # # FileMat = '200705_000_baseline.mat'
    # # FileMat = '200724_000_misting_vs_NO.mat'
    # # Folder = '200424_B6J_Misting_NO/'
    # # FileMat = '200526_B6J_misting_NO.mat'
    # # Folder = '200702_B6J_BurrowingSD/'
    # # FileMat = '200724_000_B6J_burrowingSD.mat'
    # #
    # # Folder = '200925_B6J_Burrowing_Misting_NO/'
    # # FileMat = '201019_000_B6J_Burrowing_Misting_NO.mat'
    #
    # # Folder = '201208_MCH_DREADDs/Ephys/'
    # # File = '201214_000.smrx'
    # # File = '210120_000.smrx'
    # # File = '210121_000.smrx'
    # # File = '210122_000.smrx'
    # #
    # # Folder = '191028_MCHCre_hM4Di_antistatic/'
    # # File = '191122_002.smrx'
    # #
    # Folder = '201016_B6J_NO_misting_burrowing_flouring/'
    # # File = '201025_000.smrx'
    # File = '201110_000_B6J_4SD.smrx'
    # # File = '201207_000_B6J_SD3andSN6_NO.smrx'
    # #
    # # Folder = '201022_Nms_NO_misting_burrowing/Ephys/'
    # # # File = '201030_000.smrx'
    # # File = '201127_000_SD_duringLD.smrx'
    # # File = '201109_000_baseline_NMS.smrx'
    # # File ='201207_000_SD_duringDD.smrx'
    # #
    # # Folder = '210125_electrode_placements/Ephys/'
    # # File = '210130_000.smrx'
    # # File = '210201_000_tin_strips_2_5_7.smrx'
    #
    # # Folder = '210217_EEG_electrode_placements_2/Ephys/'
    # # File = '210226_000_EEG_placements2.smrx'
    #
    # # Folder ='Smad1_PVcre_project/210211_Smad1_PVcre/Ephys/'
    # # File = '210211_Smad1_PVcre.smrx'
    #
    # # Folder = '210202_Galcre_DREADD_NO_SD/Ephys/'
    # # File = '210223_000_Galncre_DREADD.smrx'
    #
    # Folder = '210216_Vglut2Cre_SuM_AAVhM3/Ephys/'
    # File = '210301_000_ZT3_0.5mgkg_CNO.smrx'
    #
    # Folder = '210330_Vglut2-Cre_SuM_AAV-hM3_repeat_3/Ephys/'
    # File = '210417_000_0.3mgkg_CNO_0.1mgkg_CNO.smrx'
    # # File = '210502_000_ZT0_ZT12_0.3mgkg_CNO.smrx'
    #
    # Folder = "210323_Vglut2-Cre_SuM_AAV-hM3_repeat_2/Ephys/"
    # File = "210502_000_ZT0_ZT12_0.3mgkg_CNO.smrx"
    # File = '210521_000_ZT0_iDisco.smrx'
    # # File = "210421_000_0.3mgkg_CNO.smrx"
    # File = "210429_000_0.1mgkg_CNO.smrx"
    # # File = "210502_000_ZT0_ZT12_0.3mgkg_CNO.smrx"
    #
    Folder = '210409_White_noise/Ephys/'
    File = '210409_000.smrx'

    # Folder = '210419_Vglut2-Cre_SuM_AAV-hM3_repeat_4/Ephys/'
    # File = '210525_000.smrx'
    #
    # Folder = "210526_Sert-Cre_CS_hM3_round_1/Ephys/"
    # File = '210629_000_injections_ZT12.smrx'
    # File = '210708_000_injections_ZT0.smrx'

    # Folder = "210609_Vglut2-Cre_SuM_AAV-hM3_females_round2/Ephys/"
    # File = "210717_000.smrx"

    # Folder = "210527_Vglut2-Cre_CS_hM3, round1/Ephys/"
    # File = "210715_000.smrx"
    # Folder = "210726_B6Jv_circadian_fos/rig2/Ephys/"
    # File = "210830_000.smrx"
    Folder = "Smad1_PVcre_project/210823_Smad1_PVcre/Ephys/"
    File = "210909_000.smrx"

    # Folder = '210510_Vglut2-Cre_SuM_AAV-hM3_females_round1/Ephys/'
    # File = '210624_000_CNO_ZT3.smrx'

    FilePath = EphysDir+Folder+File


    # # ALL mice
    # #Create directory to save figures
    # figureFolder = mh.gen_folder(EphysDir, Folder,all_mice=True)
    # mh.figure_tail = '_'+mh.genotype +'.png'
    # ONE mouse
    #Create directory to save figures
    figureFolder = mh.gen_folder(EphysDir,Folder)
    mh.figure_tail = ' - {} - {}.png'.format(mh.pos, mh.genotype)
    # # Load data
    # mh.add_data(EphysDir + Folder,FileMat)
    mh.read_smrx(FilePath)
    # standard functions for plotting
    plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
    # Set figure resolution
    dpi = 500
    # -----------------------------------------------------------------

    ###------------------------
    #OPTIONAL remove drifting baseline
    #This might be making 60Hz noise worse, so it needs to be called on demand
    # _ , mh.EEG_data = iterative_savitzky_golay(mh,iterations=3)


    #### -------------------
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

    # def shortprocessing():
    mh.multitaper(resolution=2)
    mh.process_spectrum(smooth_iter=4,window_size=41)