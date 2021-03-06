import os

EphysDir = 'D:/Project_Mouse/Ongoing_analysis/'

Folder = '211011_Sert-Cre_Ai14_CS_Casp3/Ephys/'
File = '211102_000.smrx'

ANNfolder = 'D:/Project_mouse/Ongoing_analysis/ANN_training/'

FilePath = EphysDir+Folder+File


# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 500

base_path = os.getcwd().replace("\\", "/") + "/"
offline_data_path = base_path + 'OfflineAnalysis/data/'
state_averages_path = offline_data_path + 'StatesAverages.pkl'
