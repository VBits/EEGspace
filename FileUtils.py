import joblib
import pickle
import numpy as np
import os
import h5py
import Config
from tslearn.preprocessing import TimeSeriesResampler


def load_or_recreate_file(path, recreate_function, recreate_file=False):
    if recreate_file or not os.path.isfile(path):
        object = recreate_function()
        dump_with_correct_lib(path, object)
    else:
        object = load_with_correct_lib(path)
    return object


def dump_with_correct_lib(path, object):
    f = open(path, 'wb')
    if path.endswith(".pkl"):
        pickle.dump(object, f)
    if path.endswith(".joblib"):
        joblib.dump(object, f)
    if path.endswith(".npy"):
        np.array(object).dump(path)


def load_with_correct_lib(path):
    f = open(path, 'rb')
    if path.endswith(".pkl"):
        return pickle.load(f)
    if path.endswith(".joblib"):
        return joblib.load(f)
    if path.endswith(".npy"):
        return np.load(f, allow_pickle=True)


def get_raw_data_and_downsample(mouse_num, f=None):
    if f is None:
        f = h5py.File(Config.raw_data_file, 'r')
    ch_name = list(f.keys())
    mouse_ch = [s for s in ch_name if "G{}".format(mouse_num) in s]
    fs = 1 / f["{}".format(mouse_ch[0])]['interval'][0][0]
    downsample_rate = fs / Config.eeg_fs
    eeg_data = f[str(mouse_ch[0])]["values"][0, :]
    new_size = int(len(eeg_data) // downsample_rate)
    eeg_data = TimeSeriesResampler(sz=new_size).fit_transform(eeg_data).flatten()
    return eeg_data
