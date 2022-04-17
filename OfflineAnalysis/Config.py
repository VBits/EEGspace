import os

base_path = 'D:/Project_Mouse/Ongoing_analysis/'
experimental_path = '220113_VgatCre_CS_casp3_round1/Ephys/'
file = '220223_000.smrx'
mouse_description = 'VgatCre_CS_YFP'
mouse_id = '13'
experiment_id = experimental_path[:6]
file_id = file[:6]
load_previously_analyzed_data = True

target_fs=100
epoch_seconds = 2
smoothing_window = 21
random_epoch_size = 20000
use_ann= False
save_figures = False
lda_components = 3
dpa_z=0.9
dpa_k_max=201
knn_n_neighbors = 201



lda_figure_title_no_labels = 'LDA no labels m{}-{}_{}'.format(mouse_id, experiment_id, file_id)
lda_figure_title_dpc_labels = 'LDA DPC labels m{}-{}_{}'.format(mouse_id, experiment_id, file_id)
lda_figure_title_state_labels = 'LDA state labels m{}-{}_{}'.format(mouse_id, experiment_id, file_id)

eeg_figure_title = 'EEG m{}-{}_{} '.format(mouse_id, experiment_id, file_id, )

ann_folder = 'D:/Project_mouse/Ongoing_analysis/ANN_training/'


# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 500

base_path = os.getcwd().replace("\\", "/") + "/"
base_path = 'C:/Users/bitsik0000/PycharmProjects/ClosedLoopEEG'
offline_data_path = base_path + '/OfflineAnalysis/data/'
state_averages_path = offline_data_path + 'StateAverages.pkl'


knn_file = offline_data_path + 'knn_average.joblib'

state_df_filename = 'states_{}_{}_{}_m{}.pkl'.format(experiment_id, file_id, mouse_description, mouse_id)
lda_filename = base_path + experimental_path + 'lda_{}_{}_{}_m{}.joblib'.format(experiment_id, file_id, mouse_description, mouse_id)
knn_filename = base_path + experimental_path + 'knn_{}_{}_{}_m{}.joblib'.format(experiment_id, file_id, mouse_description, mouse_id)