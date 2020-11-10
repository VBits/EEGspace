from random import *
import struct
import Config
import threading
import time
import os
import numpy as np
import pickle
import pandas as pd
import scipy
import Preprocessing
from Timer import Timer
import h5py
import matplotlib.pyplot as plt

def get_data(mouse_num):
    f = h5py.File(Config.raw_data_file, 'r')
    ch_name = list(f.keys())
    mouse_ch = [s for s in ch_name if "G{}".format(mouse_num) in s]
    eeg_data = f[str(mouse_ch[0])]["values"][0, :]
    eeg_data = scipy.signal.resample(eeg_data, int(len(eeg_data) / 2.5))
    f = open(Config.raw_data_pkl_file, "wb")
    pickle.dump({"eeg_data": eeg_data}, f)
    return eeg_data


def cycle_test_files(file_lock, use_random=False):
    f = open(Config.raw_data_pkl_file, "rb")
    eeg_data = pickle.load(f)["eeg_data"]
    #used for testing preprocessing method
    #Preprocessing.transform_data(eeg_data, Timer("start_time", 0, 0))
    epoch_size = Config.num_seconds_per_epoch * Config.eeg_fs
    for i in range(0, Config.num_channels):
        path = Config.channel_file_base_path.format(channel_number=i)
        with file_lock:
            if os.path.isfile(path):
                os.remove(path)
        if use_random:
            random_sample_size = epoch_size * 100
            random_index = randint(0, len(eeg_data) - random_sample_size)
            random_sample = eeg_data[random_index:random_index + random_sample_size]
            threading.Thread(target=file_creation, args=(random_sample, epoch_size, i, file_lock)).start()
        else:
            threading.Thread(target=file_creation, args=(eeg_data, epoch_size, i, file_lock)).start()


def file_creation(data_points, epoch_size, channel_number, file_lock):
    start_time = 0
    path = Config.channel_file_base_path.format(channel_number=channel_number)
    for epoch in range(0, int(len(data_points) / epoch_size)):
        while os.path.isfile(path):
            continue
        with file_lock:
            with open(Config.channel_file_base_path.format(channel_number=channel_number), "wb") as f:
                f.write(struct.pack('<d', np.float64(start_time + (epoch * Config.num_seconds_per_epoch))))
                f.write(struct.pack('<i', epoch_size))
                start_point = epoch * epoch_size
                for data_point in data_points[start_point:start_point + epoch_size]:
                    f.write(struct.pack('<f', data_point))
        time.sleep(Config.num_seconds_per_epoch - ((time.time() - start_time) % Config.num_seconds_per_epoch))

