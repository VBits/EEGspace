import os

BaseDir = 'D:/SleepHomeostasis_D/'


ExpDir = '260713_VgatSert_TeNT_MR/ephys/'
File = '260714_000.smrx'


ANNfolder = 'D:/Project_mouse/Ongoing_analysis/ANN_training/'


# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
lda_figure_title_no_labels = "LDA_NoLabels"
lda_figure_title_state_labels = "LDA_StateLabels"
lda_figure_title_dpc_labels = "LDA_DPCLabels"
dpi = 300
# Set figure resolution
dpi = 300

base_path = os.getcwd().replace("\\", "/") + "/"
base_path = 'C:/Users/joo0000/PycharmProjects/ClosedLoopEEG'
offline_data_path = base_path + '/OfflineAnalysis/data/'
state_averages_path = offline_data_path + 'StateAverages.pkl'
