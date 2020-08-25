import pandas
import struct
import numpy as np
import threading
import os
import Config
import utils
import time
import matplotlib.pyplot as plt

def read_files(fs, file_lock, mouse_to_compare):
    for i in range(0, Config.num_channels):
        run_loop(fs, i, file_lock, mouse_to_compare)
        #threading.Thread(target=run_loop, args=(fs, i,)).start()

def run_loop(fs, channel_number, file_lock, mouse_to_compare):
    epoch_count = 0
    epoch_size = Config.num_seconds_per_epoch * fs
    data_points = []
    time_points = []
    total_points = 0
    while True:
        #set timer
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

            mh.PCA(normalizer=False, robust=True, scaler=mouse_to_compare.scaler)
            X = mh.pC
            cats1 = []
            cats2 = []
            for i, point in enumerate(X):
                x = point[0]
                y = point[1]
                point2 = mouse_to_compare.pC[i]
                x1 = point2[0]
                y1 = point2[0]
                print("difference of xs: " + str(x - x1))
                print("difference of ys: " + str(y - y1))
                sws = is_slow_wave(x, y)
                wake = is_wake(x, y)
                rem = is_rem(x, y)
                cat1 = ("sws" if sws else "wake" if wake else "rem" if rem else "ambiguous")
                print(cat1 + " for point " + str(i+1))
                cats1.append(cat1)
                sws1 = is_slow_wave(x1, y1)
                wake1 = is_wake(x1, y1)
                rem1 = is_rem(x1, y1)
                cat2 = ("sws" if sws1 else "wake" if wake1 else "rem" if rem1 else "ambiguous")
                print(cat2 + " for point " + str(i + 1) + " for OG data")
                cats2.append(cat2)

            data_points = data_points[epoch_size:]
            # plt.scatter(*mh.pC.T[:, :], c='k', linewidths=0, alpha=0.4, s=4)
            # plt.title('PCA for read in data')
            # plt.show()
            # plt.scatter(*mouse_to_compare.pC.T[:, 0:epoch_count], c='k', linewidths=0, alpha=0.4, s=4)
            # plt.title('PCA for saved data')
            # plt.show()\
            # mh2 = utils.Mouse("TRAP", 2)
            # mh2.EEG_fs = 100
            # mh2.EEG_data = np.array(mouse_to_compare.EEG_data[8400:16800])
            # mh2.start = [pandas.to_datetime('today')]
            # nperseg = 4 * mh2.EEG_fs
            # mh2.sleep_bandpower(nperseg=nperseg, fs=mh2.EEG_fs, EMG=False, LP_filter=True, iterations=1, mode="mirror")
            # mh2.PCA(normalizer=False, robust=True, scaler=mouse_to_compare.scaler)
            print(cats1)
            print(cats2)
            num = 6
            new_values = mh.df.T.head(num)
            for i in range(0, num):
                plt.plot(mh.x[i, :], color="red")
                #plt.plot(new_values.values[i][:42], color="red")
            # for i in range(0, num):
            #     plt.plot(mh2.x[i, :], color="green")
            old_values = mouse_to_compare.df.T.head(6)
            for i in range(0, num):
                plt.plot(mouse_to_compare.x[i, :], color="blue")
                #plt.plot(old_values.values[i][:42], color="blue")
            plt.title('PCA data')
            plt.show()
            print("stop time")

        epoch_count = epoch_count + 1
        #end timer
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

