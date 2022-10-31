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
    f = Storage.read_smrx(Config.raw_data_file) #don't load file multiple times
    for mouse_id in Config.mouse_ids:
        channel_number = Config.get_channel_number_from_mouse_id(mouse_id)
        path = Config.channel_file_base_path.format(channel_number=channel_number)
        eeg_data = Storage.get_smrx_data(f, mouse_id, Config.downsample_fs)
        with file_lock:
            if os.path.isfile(path):
                os.remove(path)
        if use_random:
            random_sample_size = epoch_size * 100
            random_index = randint(0, len(eeg_data) - random_sample_size)
            random_sample = eeg_data[random_index:random_index + random_sample_size]
            thread = threading.Thread(target=file_creation, args=(random_sample, epoch_size, channel_number, file_lock))
        else:
            thread = threading.Thread(target=file_creation, args=(eeg_data, epoch_size, channel_number, file_lock))
        thread.setDaemon(True)
        thread.start()


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

