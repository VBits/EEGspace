import os

BaseDir = 'D:/Project_Mouse/Ongoing_analysis/'


ExpDir = '220113_VgatCre_CS_casp3_round1/Ephys/'
# File = '211221_000.smrx'
File = '220223_000.smrx'


ANNfolder = 'D:/Project_mouse/Ongoing_analysis/ANN_training/'


# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 500

base_path = os.getcwd().replace("\\", "/") + "/"
base_path = 'C:/Users/bitsik0000/PycharmProjects/ClosedLoopEEG'
offline_data_path = base_path + '/OfflineAnalysis/data/'
state_averages_path = offline_data_path + 'StateAverages.pkl'
