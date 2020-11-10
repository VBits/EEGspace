import Preprocessing
from Timer import Timer
import Config
import pickle
import h5py
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_data(mouse_num):
    f = h5py.File(Config.raw_data_file, 'r')
    ch_name = list(f.keys())
    mouse_ch = [s for s in ch_name if "G{}".format(mouse_num) in s]
    eeg_data = f[str(mouse_ch[0])]["values"][0, :]
    eeg_data = scipy.signal.resample(eeg_data, int(len(eeg_data) / 2.5))#todo is this causing you issues?
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
    timer = Timer("start_time", 0, 0) #making things reliant on timer might not have been the smartest idea
    eeg_data = get_data(mouse_num)
    preconverted_data = get_preconverted_data(mouse_num)
    states = get_states(mouse_num)
    multitaper_df = Preprocessing.apply_multitaper(eeg_data)
    sxx_df = Preprocessing.do_smoothing(multitaper_df, timer)
    norm = Preprocessing.calculate_norm(sxx_df, multitaper_df.columns, timer)
    sxx_norm = sxx_df.add(norm, axis=0)
    sxx_norm = pd.DataFrame(data=sxx_norm.T.values,columns=multitaper_df.columns, index=multitaper_df.index)
    save_data(mouse_num, eeg_data, sxx_norm, preconverted_data, states, norm)
    #todo test the accuracy of the classication as well
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

#create_mouse_data_object()

load_mouse_data_object()





