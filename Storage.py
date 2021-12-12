"""
Online analysis
"""
import joblib
import pickle
import numpy as np
import os
import h5py
from OnlineAnalysis import Config
from tslearn.preprocessing import TimeSeriesResampler
import pandas as pd
import threading
import struct
from sonpy import lib as sp
from math import floor
from scipy.signal import decimate, butter, dlti


def load_or_recreate_file(path, recreate_function, recreate_file=False):
    if recreate_file or not os.path.isfile(path):
        object = recreate_function()
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


def consume_spike2_output_data_file(path):
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


def load_downsampled_raw_data(mouse_num, f=None):
    if f is None:
        f = read_smrx(Config.raw_data_file)
    ch_name = list(f.keys())
    mouse_ch = [s for s in ch_name if "G{}".format(mouse_num) in s]
    fs = 1 / f["{}".format(mouse_ch[0])]['interval'][0][0]
    downsample_rate = fs / Config.eeg_fs
    eeg_data = f[str(mouse_ch[0])]["values"][0, :]
    new_size = int(len(eeg_data) // downsample_rate)
    eeg_data = TimeSeriesResampler(sz=new_size).fit_transform(eeg_data).flatten()
    return eeg_data

def downsample_EGG(EEG_data, EEG_fs, target_fs=100):
    '''
    Downsample the data to a target frequency of 100Hz

    You can also replace the Butterworth filter with Bessel filter or the default Chebyshev filter.
    system = dlti(*bessel(4,0.99))
    system = dlti(*cheby1(3,0.05,0.99))
    All filters produced very similar results for downsampling from 200Hz to 100Hz
    '''
    EEG_fs = round(EEG_fs)
    rate = EEG_fs/ target_fs
    system = dlti(*butter(4,0.99))
    EEG_data = decimate(EEG_data, round(rate), ftype=system, zero_phase=True)
    EEG_fs = EEG_fs / rate
    return EEG_data, EEG_fs


def read_smrx(file_path):
    file = sp.SonFile(file_path, True)

    if file.GetOpenError() != 0:
        print('Error opening file:', sp.GetErrorString(file.GetOpenError()))
        quit()
    return file


def get_downsampled_smrx_data(file, mouse_num, target_fs):
    wave_chan = mouse_num - 1

    dMaxSeconds = file.ChannelMaxTime(wave_chan) * file.GetTimeBase()

    dPeriod = file.ChannelDivide(wave_chan) * file.GetTimeBase()
    nPoints = floor(dMaxSeconds / dPeriod)

    EEG_data = np.array(file.ReadFloats(wave_chan, nPoints, 0))

    EEG_fs = 1 / dPeriod

    downsampled, _ = downsample_EGG(EEG_data, EEG_fs, target_fs)
    return downsampled
