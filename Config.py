import os

# numerical settings
eeg_fs = 100
smoothing_iterations = 4
num_channels = 1
num_seconds_per_epoch = 2
iteration_buffer = 100
mice_numbers = [2]  # [1, 2, 3, 4, 5, 6, 7]  # array of numbers 1-7
print_timer_info_for_mice = [2]#[1, 2, 3, 4, 5, 6, 7]

# boolean
cycle_test_data = False
recreate_model_file = True
recreate_lda = False
recreate_knn = False

# file paths and file names
base_path = os.getcwd().replace("\\", "/") + "/"
channel_file_base_path = base_path + "ChannelData/{channel_number}/data.bin" if cycle_test_data \
    else "C:/Users/bitsik0000/SleepData/binaries/{channel_number}/data.bin"
data_path = base_path + "data/"
mouse_object_path = data_path + "mouse_object.pkl"
lda_model_path = data_path + "model.pkl"
mouse_model_path = data_path + "Models/{mouse_num}/mouse_model.pkl"
training_data_path = data_path + "CombinedData/"
raw_data_file = training_data_path + "200724_000_B6J_burrowingSD.mat"
raw_data_pkl_file = data_path + "raw_test_data.pkl"
multitaper_data_file_path = training_data_path + 'Sxx_norm_200702_B6J_m{mouse_num}.pkl'
state_file_path = training_data_path + 'states_200702_B6J_m{mouse_num}.pkl'
lda_file_path = training_data_path + 'lda_200702_m{mouse_num}.joblib'
knn_file_path = training_data_path + 'knn_200702_m{mouse_num}.joblib'
norm_file_path = training_data_path + 'norm_200702_B6J_m{mouse_num}.npy'
