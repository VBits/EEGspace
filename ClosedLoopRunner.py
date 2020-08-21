import pandas
import struct
import numpy as np
import threading
import os
import Config
import utils
import time

def read_files(fs, file_lock):
    for i in range(0, Config.num_channels):
        run_loop(fs, i, file_lock)
        #threading.Thread(target=run_loop, args=(fs, i,)).start()

def run_loop(fs, channel_number, file_lock):
    epoch_count = 0
    epoch_size = Config.num_seconds_per_epoch * fs
    data_points = []
    time_points = []
    total_points = 0
    #last_created = None
    while True:
        path = Config.channel_file_base_path.format(channel_number=channel_number)
        if not os.path.isfile(path):
            continue
        # created = time.ctime(os.path.getctime(path))
        # if created == last_created:
        #     continue
        # last_created = created
        with file_lock:
            with open(path, "rb") as f:
                bytes_read = f.read(8)
                time_points.append(struct.unpack('<d', bytes_read)[0])
                bytes_read = f.read(4)
                total_points = total_points + struct.unpack('<i', bytes_read)[0]
                bytes_read = f.read(4)
                while bytes_read:
                    data_points.append(struct.unpack('<f', bytes_read)[0])
                    bytes_read = f.read(4)
            os.remove(path)

        if epoch_count > 9:
            mh = utils.Mouse("TRAP", 2)
            mh.EEG_fs = 100
            mh.EEG_data = np.array(data_points)
            mh.start = [pandas.to_datetime('today')]
            nperseg = 4 * mh.EEG_fs
            mh.sleep_bandpower(nperseg=nperseg, fs=mh.EEG_fs, EMG=False, LP_filter=True, iterations=1)

            mh.PCA(normalizer=False, robust=True)
            X = mh.pC
        epoch_count = epoch_count + 1


def f1(x):
    return 30 * x - 6.5


def f2(x):
    return x - 0.25


def is_slow_wave(x, y):
    return f1(x) <= y <= f2(x)


def is_wake(x, y):
    return f1(x) >= y >= f2(x)


def is_rem(x, y):
    return y >= f1(x) and y >= f2(x)