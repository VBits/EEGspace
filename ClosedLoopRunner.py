import pandas
import struct
import numpy as np
import threading
import os
import Config
import utils
import time

def read_files(fs, file_lock, training_mouse_object):
    for i in range(0, Config.num_channels):
        run_loop(fs, i, file_lock, training_mouse_object)
        #threading.Thread(target=run_loop, args=(fs, i,)).start()

def run_loop(fs, channel_number, file_lock, training_mouse_object):
    epoch_count = 0
    epoch_size = Config.num_seconds_per_epoch * fs
    data_points = []
    time_points = []
    total_points = 0
    while True:
        start_time = time.perf_counter()
        path = Config.channel_file_base_path.format(channel_number=channel_number)
        if not os.path.isfile(path):
            continue
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

        if epoch_count > 41:
            mh = utils.Mouse("TRAP", 2)
            mh.EEG_fs = 100
            mh.EEG_data = np.array(data_points)
            mh.start = [pandas.to_datetime('today')]
            nperseg = 4 * mh.EEG_fs
            mh.sleep_bandpower(nperseg=nperseg, fs=mh.EEG_fs, EMG=False, LP_filter=True, iterations=1, mode="mirror")

            mh.PCA(normalizer=False, robust=True, scaler=training_mouse_object.scaler, saved_pca=training_mouse_object.pca)
            X = mh.pC
            for i, point in enumerate(X):
                x = point[0]
                y = point[1]
                sws = is_slow_wave(x, y)
                wake = is_wake(x, y)
                rem = is_rem(x, y)
                cat = ("sws" if sws else "wake" if wake else "rem" if rem else "ambiguous")
                print(cat + " for point " + str(i+1))

            data_points = data_points[epoch_size:]

        epoch_count = epoch_count + 1
        end_time = time.perf_counter()
        print("time for iteration was: " + str(end_time - start_time))


def f1(x):
    return -5 * x - 1


def f2(x):
    return 0.8 * x + 0.25


def is_slow_wave(x, y):
    return f1(x) <= y <= f2(x)


def is_rem(x, y):
    return y >= f1(x) and y >= f2(x)


def is_wake(x, y):
    return y <= f1(x)

