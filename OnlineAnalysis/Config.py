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
median_filter_buffer = 21
median_filter_buffer_middle = math.ceil(median_filter_buffer / 2)

#rig position
mouse_ids = [1]
mouse_id_to_channel_mapping = {1: 0}
#this maps the mouse id to a channel which maps to the bin file location
#e.g. mouse 13 is outputting files to bin folder 7 = mouse_id_to_channel_mapping = {13: 7}
#the mouse id will be used as the default if a mapping is not specified.
print_timer_info_for_mice = [1]#[1, 2, 3, 4, 5, 6, 7]
comport = "COM5"

# boolean
cycle_test_data = False
recreate_model_file = True
recreate_lda = True
recreate_knn = True

# file paths and file names
base_path = os.getcwd().replace("\\", "/") + "/"
#base_path = "D:/temp_Matt/"
channel_file_base_path = \
    base_path + "ChannelData/{channel_number}/data.bin" if cycle_test_data \
    else "C:/Users/bitsik0000/SleepData/binaries/{channel_number}/data.bin"

#Naming and file convensions
data_path = base_path + "data/"
training_data_path = data_path + "CombinedData/"
raw_data_file = training_data_path + "220202_000.smrx"
run_name = "220107_220202_Gad2Cre_CS_hM3"
multitaper_data_file_path = training_data_path + 'Multitaper_df_'+run_name+'_m{mouse_id}.pkl'
combined_data_file_path = training_data_path + 'Multitaper_df_'+run_name+'_combined_with_medians_m{mouse_id}.pkl'
state_file_path = training_data_path + 'states_'+run_name+'_m{mouse_id}.pkl'
lda_file_path = training_data_path + 'lda_'+run_name+'_m{mouse_id}.joblib'
knn_file_path = training_data_path + 'knn_'+run_name+'_m{mouse_id}.joblib'

state_colors = {
    "REM": "#443e99",
    "SWS": "#3e8399",
    "LMwake": "#3e9944",
    "HMwake": "#f2ef30",
    "LTwake": "#3e9944",
    "HTwake": "#f2ef30",
}

def get_channel_number_from_mouse_id(mouse_id):
    if mouse_id in mouse_id_to_channel_mapping:
        return mouse_id_to_channel_mapping[mouse_id]
    return mouse_id

