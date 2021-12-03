import os

EphysDir = 'D:/Project_Mouse/Ongoing_analysis/'

Folder = '210409_White_noise/Ephys/'
File = '210409_000.smrx'
# Folder = '211014_Vglut2-Cre_CS_Casp3/Ephys/'
# Folder = '211011_Sert-Cre_Ai14_CS_Casp3/Ephys/'
# File = '211102_000.smrx'
FilePath = EphysDir+Folder+File

ANNfolder = 'D:/Project_mouse/Ongoing_analysis/ANN_training/'


# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 500

base_path = os.getcwd().replace("\\", "/") + "/"
base_path = r'C:\Users\bitsik0000\PycharmProjects\ClosedLoopEEG'
offline_data_path = base_path + '/OfflineAnalysis/data/'
state_averages_path = offline_data_path + 'StateAverages.pkl'
