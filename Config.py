import os
eeg_fs = 100
smoothing_iterations = 1
num_channels = 1
num_seconds_per_epoch = 2
iteration_buffer = 100
cycle_test_data = True
recreate_model_file = False
base_path = os.getcwd().replace("\\", "/") + "/"
channel_file_base_path = base_path + "ChannelData/{channel_number}/data.bin" \
    if cycle_test_data else "C:/Users/bitsik0000/SleepData/binaries/{channel_number}/data.bin"
data_path = base_path + "data/"
mouse_object_path = data_path + "mouse_object.pkl"
lda_model_path = data_path + "model.pkl"
training_data_path = data_path + "Multitaper_df_norm/"
raw_data_file = data_path + "181008_000_baseline.mat"
raw_data_pkl_file = data_path + "raw_test_data.pkl"
print_timer_info_for_mice = [0]

