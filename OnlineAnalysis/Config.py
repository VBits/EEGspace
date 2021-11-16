"""
Inputs defined by user
Online analysis
"""

import os

#Digitization frequency
eeg_fs = 200
downsample_fs = 100
#Specify the time resolution for each epoch
num_seconds_per_epoch = 2
#Specify the buffer length
epoch_buffer = 50
#Savgol filtering parameters
savgol_window=41
savgol_order=2
#Savgol iterations (recommended range 0-4)
smoothing_iterations = 4
#Specify number of epochs used for dimensionality reduction
LDA_epochs = 51
#rig position
rig_position = [1]  # array of numbers 1-8
print_timer_info_for_mice = [1]#[1, 2, 3, 4, 5, 6, 7]

# boolean
cycle_test_data = True
recreate_model_file = True
recreate_lda = False
recreate_knn = True

# file paths and file names
base_path = os.getcwd().replace("\\", "/") + "/"
channel_file_base_path = \
    base_path + "ChannelData/{channel_number}/data.bin" if cycle_test_data \
    else "C:/Users/bitsik0000/SleepData/binaries/{channel_number}/data.bin"

#Naming and file convensions
data_path = base_path + "data/"
mouse_object_path = data_path + "mouse_object.pkl"
lda_model_path = data_path + "lda_model.pkl"
mouse_model_path = data_path + "Models/{mouse_num}/mouse_model.pkl"
training_data_path = data_path + "CombinedData/"
raw_data_file = training_data_path + "200724_000_B6J_burrowingSD.mat"
raw_data_pkl_file = data_path + "raw_test_data.pkl"
run_name = "210409_210409_B6J"
multitaper_data_file_path = training_data_path + 'Sxx_norm_'+run_name+'_m1.pkl'
state_file_path = training_data_path + 'states_'+run_name+'_m1.pkl'
lda_file_path = training_data_path + 'lda_'+run_name+'_m1.joblib'
knn_file_path = training_data_path + 'knn_'+run_name+'_m1.joblib'
norm_file_path = training_data_path + 'norm_'+run_name+'_m1.npy'

state_colors = {
    "REM": "#443e99",
    "SWS": "#3e8399",
    "LMwake": "#3e9944",
    "HMwake": "#f2ef30",
}