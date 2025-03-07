import os

BaseDir = 'D:/SleepHomeostasis_D/'


ExpDir = '250205_VgatSert_Kir21_MR/ephys/'
File = '250211_000.smrx'


ANNfolder = 'D:/Project_mouse/Ongoing_analysis/ANN_training/'


# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 300

base_path = os.getcwd().replace("\\", "/") + "/"
base_path = 'C:/Users/joo0000/PycharmProjects/ClosedLoopEEG'
offline_data_path = base_path + '/OfflineAnalysis/data/'
state_averages_path = offline_data_path + 'StateAverages.pkl'
