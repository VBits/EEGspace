import os

BaseDir = 'D:/Project_Mouse/Ongoing_analysis/'


ExpDir = '220113_VgatCre_CS_casp3_round1/Ephys/'
File = '220223_000.smrx'

genotype = 'VgatCre_CS_YFP'
rig_pos = '13'
load_previously_analyzed_data = True

random_epoch_size = 20000
use_ANN= False
save_figures = False
LDA_components = 3
DPA_Z=0.9
DPA_k_max=201
KNN_n_neighbors = 201



LDA_figure_title_no_labels = 'LDA no labels rig position {}-{}_{}'.format(rig_pos,ExpDir[:6],File[:6])
LDA_figure_title_DPC_labels = 'LDA DPC labels rig position {}-{}_{}'.format(rig_pos,ExpDir[:6],File[:6])
LDA_figure_title_state_labels = 'LDA state labels rig position {}-{}_{}'.format(rig_pos,ExpDir[:6],File[:6])

EEG_figure_title = 'EEG rig position {}-{}_{} '.format(rig_pos,ExpDir[:6],File[:6],)

ANNfolder = 'D:/Project_mouse/Ongoing_analysis/ANN_training/'


# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 500

base_path = os.getcwd().replace("\\", "/") + "/"
base_path = 'C:/Users/bitsik0000/PycharmProjects/ClosedLoopEEG'
offline_data_path = base_path + '/OfflineAnalysis/data/'
state_averages_path = offline_data_path + 'StateAverages.pkl'


knn_file = offline_data_path + 'knn_average.joblib'
