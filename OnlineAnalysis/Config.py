"""
Inputs defined by user
Online analysis
"""
import math
import os

#Digitization frequency
eeg_fs = 200
downsample_fs = 100
#Specify the time resolution for each epoch
num_seconds_per_epoch = 2
#Specify the buffer length
epoch_buffer = 50
median_filter_buffer = 21
median_filter_buffer_middle = math.ceil(median_filter_buffer / 2)

#Savgol iterations (recommended range 0-4)
smoothing_iterations = 4
#Specify number of epochs used for dimensionality reduction
LDA_epochs = 51
#rig position
rig_position = [1]  # array of numbers 1-8
print_timer_info_for_mice = [1]#[1, 2, 3, 4, 5, 6, 7]
comport = "COM5"

# boolean
cycle_test_data = True
recreate_model_file = True
recreate_lda = True
recreate_knn = True

# file paths and file names
# base_path = os.getcwd().replace("\\", "/") + "/"
base_path = "D:/temp_Matt/"
channel_file_base_path = \
    base_path + "ChannelData/{channel_number}/data.bin" if cycle_test_data \
    else "C:/Users/bitsik0000/SleepData/binaries/{channel_number}/data.bin"

#Naming and file convensions
data_path = base_path + "data/"
mouse_object_path = data_path + "mouse_object.pkl"
lda_model_path = data_path + "lda_model.pkl"
mouse_model_path = data_path + "Models/{mouse_num}/mouse_model.pkl"
training_data_path = data_path + "CombinedData/"
raw_data_file = training_data_path + "220202_000.smrx"
raw_data_pkl_file = data_path + "raw_test_data.pkl"
run_name = "220107_220202_Gad2Cre_CS_hM3"
multitaper_data_file_path = training_data_path + 'Multitaper_df_'+run_name+'_m{mouse_num}.pkl'
combined_data_file_path = training_data_path + 'Multitaper_df_'+run_name+'_combined_with_medians_m{mouse_num}.pkl'
state_file_path = training_data_path + 'states_'+run_name+'_m{mouse_num}.pkl'
lda_file_path = training_data_path + 'lda_'+run_name+'_m{mouse_num}.joblib'
knn_file_path = training_data_path + 'knn_'+run_name+'_m{mouse_num}.joblib'

state_colors = {
    "REM": "#443e99",
    "SWS": "#3e8399",
    "LMwake": "#3e9944",
    "HMwake": "#f2ef30",
    "LTwake": "#3e9944",
    "HTwake": "#f2ef30",
}