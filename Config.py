import os

# numerical settings
eeg_fs = 100
smoothing_iterations = 1
num_channels = 1
num_seconds_per_epoch = 2
iteration_buffer = 100
mice_numbers = [1, 2, 3, 4, 5, 6, 7]  # array of numbers 1-8
print_timer_info_for_mice = [1]

# boolean
cycle_test_data = False
recreate_model_file = True
recreate_lda = False
recreate_knn = False

# file paths and file names
base_path = os.getcwd().replace("\\", "/") + "/"
channel_file_base_path = base_path + "ChannelData/{channel_number}/data.bin" \
    if cycle_test_data else "C:/Users/bitsik0000/SleepData/binaries/{channel_number}/data.bin"
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

# todo delete old_
old_training_data_path = data_path + "CombinedData/"
old_raw_data_file = data_path + "181008_000_baseline.mat"
old_mouse_files = ['Sxx_norm_200604_m1.pkl', 'Sxx_norm_200604_m2.pkl', 'Sxx_norm_200604_m3.pkl',
                   'Sxx_norm_200604_m4.pkl', 'Sxx_norm_200424_m5.pkl', 'Sxx_norm_200424_m6.pkl',
                   'Sxx_norm_200424_m7.pkl']
old_mouse_state_files = ['states_200604_m1.pkl', 'states_200604_m2.pkl', 'states_200604_m3.pkl', 'states_200604_m4.pkl',
                         'states_200424_m5.pkl', 'states_200424_m6.pkl', 'states_200424_m7.pkl']
