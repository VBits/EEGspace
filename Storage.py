import joblib
import pickle
import numpy as np
import os
# import h5py
# import Config
# from tslearn.preprocessing import TimeSeriesResampler
import pandas as pd
import threading
import struct


def load_or_recreate_file(path, recreate_function, recreate_file=False, params={}):
    if recreate_file or not os.path.isfile(path):
        object = recreate_function(**params)
        dump_to_file(path, object)
    else:
        object = load_from_file(path)
    return object


def dump_to_file(path, object):
    f = open(path, 'wb')
    if path.endswith(".pkl"):
        pickle.dump(object, f)
    if path.endswith(".joblib"):
        joblib.dump(object, f)
    if path.endswith(".npy"):
        np.array(object).dump(path)
    if path.endswith(".h5"):
        pd.DataFrame(object).to_hdf(path, key='df', mode='w')


def load_from_file(path):
    f = open(path, 'rb')
    if path.endswith(".pkl"):
        return pickle.load(f)
    if path.endswith(".joblib"):
        return joblib.load(f)
    if path.endswith(".npy"):
        return np.load(f, allow_pickle=True)
    if path.endswith(".h5"):
        # y = h5py.File(path, 'r')
        # z = list(y.keys())
        # test = pd.DataFrame(y[z[0]])
        x = pd.read_hdf(path, 'df')
        return x


def remove_file_if_exists(path):
    if os.path.isfile(path):
        os.remove(path)


def no_file_exists_at_location(path):
    return not os.path.isfile(path)


def consume_spike_output_data_file(path):
    if not os.path.isfile(path):
        return None, None, None
    file_lock = threading.Lock()
    data_points = []
    with file_lock:
        with open(path, "rb") as f:
            bytes_read = f.read(8)
            time_point = struct.unpack('<d', bytes_read)[0]
            bytes_read = f.read(4)
            total_points = struct.unpack('<i', bytes_read)[0]
            print(total_points)
            bytes_read = f.read(4)
            while bytes_read:
                data_points.append(struct.unpack('<f', bytes_read)[0])
                bytes_read = f.read(4)
        os.remove(path)
    return time_point, total_points, data_points


# def load_downsampled_raw_data(mouse_num, f=None):
#     if f is None:
#         f = h5py.File(Config.raw_data_file, 'r')
#     ch_name = list(f.keys())
#     mouse_ch = [s for s in ch_name if "G{}".format(mouse_num) in s]
#     fs = 1 / f["{}".format(mouse_ch[0])]['interval'][0][0]
#     downsample_rate = fs / Config.eeg_fs
#     eeg_data = f[str(mouse_ch[0])]["values"][0, :]
#     new_size = int(len(eeg_data) // downsample_rate)
#     eeg_data = TimeSeriesResampler(sz=new_size).fit_transform(eeg_data).flatten()
#     return eeg_data
