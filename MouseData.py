import Preprocessing
from Timer import Timer
import Config
import pickle
import h5py
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Model
from tslearn.preprocessing import TimeSeriesResampler
import FileUtils


def get_data(mouse_num):
    f = h5py.File(Config.raw_data_file, 'r')
    ch_name = list(f.keys())
    mouse_ch = [s for s in ch_name if "G{}".format(mouse_num) in s]
    eeg_data = f[str(mouse_ch[0])]["values"][0, :]
    eeg_data = scipy.signal.resample(eeg_data, int(len(eeg_data) / 2.5))  # todo is this causing you issues?
    return eeg_data


def get_preconverted_data(mouse_num):
    file_name = [file_name for file_name in Config.mouse_files if "_m{}".format(mouse_num) in file_name][0]
    f = open(Config.training_data_path + file_name, 'rb')
    return pickle.load(f)


def get_states(mouse_num):
    file_name = [file_name for file_name in Config.mouse_state_files if "_m{}".format(mouse_num) in file_name][0]
    f = open(Config.training_data_path + file_name, 'rb')
    return pickle.load(f)


def save_data(mouse_num, raw_data, converted, preconverted, states, norm):
    data = {
        "mouse_num": mouse_num,
        "raw_data": raw_data,
        "converted": converted,
        "preconverted": preconverted,
        "states": states,
        "norm": norm,
    }
    f = open(Config.data_path + "mouse_data_" + str(mouse_num), 'wb')
    pickle.dump(data, f)
    f = open(Config.data_path + "norm_" + str(mouse_num), 'wb')
    pickle.dump({"norm": norm}, f)


def create_mouse_data_object():
    mouse_num = 1
    timer = Timer("start_time", 0, 0)  # making things reliant on timer might not have been the smartest idea
    eeg_data = get_data(mouse_num)
    preconverted_data = get_preconverted_data(mouse_num)
    states = get_states(mouse_num)
    multitaper_df = Preprocessing.apply_multitaper(eeg_data)
    sxx_df = Preprocessing.do_smoothing(multitaper_df, timer)
    norm = Preprocessing.calculate_norm(sxx_df, multitaper_df.columns, timer)
    sxx_norm = sxx_df.add(norm, axis=0)
    sxx_norm = pd.DataFrame(data=sxx_norm.T.values, columns=multitaper_df.columns, index=multitaper_df.index)
    save_data(mouse_num, eeg_data, sxx_norm, preconverted_data, states, norm)
    # todo test the accuracy of the classication as well
    mse = np.mean((np.array(preconverted_data[0:sxx_norm.shape[0]]) - np.array(sxx_norm)) ** 2)
    print(mse)


def load_mouse_data_object():
    mouse_num = 1
    f = open(Config.data_path + "mouse_data_" + str(mouse_num), 'rb')
    data = pickle.load(f)
    n = 10  # How many digits we will display
    preconverted = np.array(data["preconverted"])
    converted = np.array(data["converted"])
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.plot(preconverted[i])
        plt.gray()

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.plot(converted[i])
        plt.gray()
    plt.show()
    print("done")

def read_in_and_resample_data(mouse_num):
    f = h5py.File(Config.raw_data_file, 'r')
    ch_name = list(f.keys())
    mouse_ch = [s for s in ch_name if "G{}".format(mouse_num) in s]
    fs = 1 / f["{}".format(mouse_ch[0])]['interval'][0][0]
    downsample_rate = fs / Config.eeg_fs
    # lenth = int(downsample_rate * 10000)
    eeg_data = f[str(mouse_ch[0])]["values"][0, :]
    new_size = int(len(eeg_data) // downsample_rate)
    og_data = eeg_data
    eeg_data = TimeSeriesResampler(sz=new_size).fit_transform(eeg_data).flatten()
    # eeg_data2 = scipy.signal.resample(og_data, new_size)

    return eeg_data


def load_raw_data(mouse_num):
    path = Config.data_path + "raw_eeg_mouse_" + str(mouse_num) + ".npy"
    return FileUtils.load_or_recreate_file(path, lambda: read_in_and_resample_data(mouse_num), recreate_file=False)


def test_conversion_using_model():
    mouse_num = 1
    timer = Timer("start_time", 0, 0)  # making things reliant on timer might not have been the smartest idea
    eeg_data = load_raw_data(mouse_num)
    model = Model.get_model_object(mouse_num)
    preconverted_data = model.training_data[0:50]
    multitaper_df = Preprocessing.apply_multitaper(eeg_data[0:10200])[0:50]
    sxx_df = Preprocessing.do_smoothing(multitaper_df, timer, 4)
    sxx_norm = sxx_df.add(model.norm, axis=0)
    sxx_norm = pd.DataFrame(data=sxx_norm.T.values, columns=multitaper_df.columns, index=multitaper_df.index)
    mse = np.mean((np.array(preconverted_data[0:sxx_norm.shape[0]]) - np.array(sxx_norm)) ** 2)
    print(mse)
    preconverted = np.array(preconverted_data)
    converted = np.array(sxx_norm)
    plt.figure(figsize=(20, 4))
    n = 10
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.plot(preconverted[i+40])
        plt.gray()

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.plot(converted[i+40])
        plt.gray()
    plt.show()
    lda_encoded = model.lda.transform(converted)
    knn_state = model.classifier.predict(lda_encoded)
    new_states = [model.states[state] for state in knn_state]
    print(new_states)
    old_knn_states = model.training_data_states[0:50]
    old_states = [model.states[state] for state in old_knn_states]
    print(old_states)
    cells_with_same_class = [i for (i, x) in enumerate(old_states) if old_states[i] == new_states[i]]
    print(len(cells_with_same_class))
    print("done")


def try_cycling_data():
    mouse_num = 1
    timer = Timer("start_time", 0, 0)  # making things reliant on timer might not have been the smartest idea
    eeg_data = load_raw_data(mouse_num)
    model = Model.get_model_object(mouse_num)
    epoch_size = Config.num_seconds_per_epoch * Config.eeg_fs
    eeg_subset = []
    mses = []
    epoch_count = 0
    preconverted_data = model.training_data
    preconverted_states = model.training_data_states
    newly_converted_states = []
    first_fifty = True
    for epoch in range(0, int(len(eeg_data) / epoch_size)):
        start_point = epoch * epoch_size
        eeg_subset.append(eeg_data[start_point:start_point + epoch_size])
        if epoch_count == 50:
            multitaper_df = Preprocessing.apply_multitaper(list(np.concatenate(eeg_subset).flat))
            sxx_df = Preprocessing.do_smoothing(multitaper_df, timer, 4)
            sxx_norm = sxx_df.add(model.norm, axis=0)
            sxx_norm = pd.DataFrame(data=sxx_norm.T.values, columns=multitaper_df.columns, index=multitaper_df.index)
            epoch_start = epoch - epoch_count
            converted_data = np.array(sxx_norm)
            mse = np.mean((np.array(preconverted_data[epoch_start:epoch]) - converted_data) ** 2)
            mses.append(mse)
            if first_fifty:
                lda_encoded = model.lda.transform(converted_data)
                first_fifty = False
            else:
                lda_encoded = model.lda.transform([converted_data[-1]])
            knn_state = model.classifier.predict(lda_encoded)
            #new_states = [model.states[state] for state in knn_state]
            newly_converted_states = newly_converted_states + list(knn_state)
            eeg_subset = eeg_subset[1:]
        else:
            epoch_count = epoch_count + 1

    same_states = [(preconverted_states[i], newly_converted_states[i]) for x, i in enumerate(newly_converted_states)
                   if preconverted_states[i] == newly_converted_states[i]]
    diff_states = [(preconverted_states[i], newly_converted_states[i]) for x, i in enumerate(newly_converted_states)
                   if preconverted_states[i] != newly_converted_states[i]]

    print("same: " + str(len(same_states)))
    print("diff: " + str(len(diff_states)))
    print("ratio: " + str(len(diff_states)/len(newly_converted_states)))

    print("done")

# create_mouse_data_object()

# load_mouse_data_object()

#test_conversion_using_model()

try_cycling_data()

print("done")
