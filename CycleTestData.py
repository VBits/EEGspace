from random import *
import struct
import Config
import threading
import time
import os
import numpy as np


def cycle_test_files(mh, file_lock):
    for i in range(0, Config.num_channels):
        epoch_size = Config.num_seconds_per_epoch * mh.EEG_fs
        random_sample_size = epoch_size * 100
        random_index = randint(0, len(mh.EEG_data) - random_sample_size)
        random_sample = mh.EEG_data[random_index:random_index + random_sample_size]
        threading.Thread(target=create_file_creation, args=(random_sample, epoch_size, i,file_lock)).start()


def create_file_creation(random_sample, epoch_size, channel_number, file_lock):
    start_time = 0
    path = Config.channel_file_base_path.format(channel_number=channel_number)
    with file_lock:
        if os.path.isfile(path):
            os.remove(path)
    for epoch in range(0, int(len(random_sample)/epoch_size)):
        while os.path.isfile(path):
            continue
        with file_lock:
            with open(Config.channel_file_base_path.format(channel_number=channel_number), "wb") as f:
                f.write(struct.pack('<d', np.float64(start_time + (epoch * Config.num_seconds_per_epoch))))
                f.write(struct.pack('<i', epoch_size))
                start_point = epoch * epoch_size
                for data_point in random_sample[start_point:start_point+epoch_size]:
                    f.write(struct.pack('<f', data_point))
        time.sleep(Config.num_seconds_per_epoch - ((time.time() - start_time) % Config.num_seconds_per_epoch))

