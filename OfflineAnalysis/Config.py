"""
experimental_path starts with an experiment ID
file starts with a file ID
We use dates that the experiment first started and a file first created as unique IDs
"""
#Update for each mouse
base_path = 'C:/Users/matthew.grant/source/repos/ClosedLoopEEG/OnlineAnalysis/data/CombinedData/'
experimental_path = ''
file = '220202_000.smrx'
mouse_description = 'Nms'
mouse_id = 1

#Update more rarely
#todo mg, update the UI to hide these most of the time
experiment_id = experimental_path[:6]
file_id = file[:6]
dpa_z=0.9
target_fs=100
epoch_seconds = 2
smoothing_window = 21
random_epoch_size = 20000
lda_components = 3
dpa_k_max=201
knn_n_neighbors = 201
quantile_norm = 0.01
eps = 2
min_samples = 100


#No need to update normally
figure_title_mouse_info = 'm{}-{}_{}'.format(mouse_id, experiment_id, file_id)
lda_figure_title_no_labels = 'LDA no labels {}'.format(figure_title_mouse_info)
lda_figure_title_dpc_labels = 'LDA DPC labels {}'.format(figure_title_mouse_info)
lda_figure_title_state_labels = 'LDA state labels {}'.format(figure_title_mouse_info)
lda_figure_title_outliers_labels = 'LDA state labels with outliers {}'.format(figure_title_mouse_info)

eeg_figure_title = 'EEG {}'.format(figure_title_mouse_info)

# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 500

# base_path = os.getcwd().replace("\\", "/") + "/"
project_path = 'C:/Users/bitsik0000/PycharmProjects/ClosedLoopEEG/OfflineAnalysis'
offline_data_path = project_path + '/data/'

average_states_path = offline_data_path + 'StateAverages.pkl'
average_knn_path = offline_data_path + 'knn_average.joblib'
average_lda_path = offline_data_path + 'lda_average.joblib' #use the version with the underscore for older versions of python

state_df_filename = base_path + experimental_path + 'states_{}_{}_{}_m{}.pkl'.format(experiment_id, file_id, mouse_description, mouse_id)
lda_filename = base_path + experimental_path + 'lda_{}_{}_{}_m{}.joblib'.format(experiment_id, file_id, mouse_description, mouse_id)
knn_filename = base_path + experimental_path + 'knn_{}_{}_{}_m{}.joblib'.format(experiment_id, file_id, mouse_description, mouse_id)