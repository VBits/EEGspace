"""
Online analysis simulation
"""

from random import *
import struct
from OnlineAnalysis import Config
import threading
import time
import os
import numpy as np
import h5py
import Storage


def cycle_test_files(file_lock, use_random=False):
    epoch_size = Config.num_seconds_per_epoch * Config.eeg_fs
    f = h5py.File(Config.raw_data_file, 'r') #don't load file multiple times
    for mouse_num in Config.rig_position:
        channel_number = mouse_num-1
        path = Config.channel_file_base_path.format(channel_number=channel_number)
        eeg_data = Storage.load_downsampled_raw_data(mouse_num, f)
        with file_lock:
            if os.path.isfile(path):
                os.remove(path)
        if use_random:
            random_sample_size = epoch_size * 100
            random_index = randint(0, len(eeg_data) - random_sample_size)
            random_sample = eeg_data[random_index:random_index + random_sample_size]
            threading.Thread(target=file_creation, args=(random_sample, epoch_size, channel_number, file_lock)).start()
        else:
            threading.Thread(target=file_creation, args=(eeg_data, epoch_size, channel_number, file_lock)).start()


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
        time.sleep(
            Config.num_seconds_per_epoch - ((time.time() - start_time) % Config.num_seconds_per_epoch))

