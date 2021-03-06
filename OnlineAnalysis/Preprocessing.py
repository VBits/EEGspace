"""
Offline analysis
"""

import nitime.algorithms as tsa
from scipy.signal import detrend, dlti, butter, decimate
from OnlineAnalysis import Config
import numpy as np
import pandas as pd
import scipy

def smooth_prev_epochs_with_savgol(series, buffer, subset_size):
    epoch_size = series.shape[1]
    subset = np.array(series[:subset_size])

    new_series = []
    subset_flattened = subset.flatten()
    for i in range(0, subset_size):
        if i < buffer:
            continue
        buffer_expanded = buffer * epoch_size
        index_expanded = i * epoch_size
        subseries = np.array(subset_flattened[index_expanded - buffer_expanded:index_expanded])
        subseries_smoothed = apply_savgol(subseries, 41, 4)
        new_series.append(subseries_smoothed)

    new_series = np.array(new_series).flatten().reshape(subset_size - buffer, epoch_size * buffer)
    return new_series

def transform_data(data_points, timer):
    timer.set_time_point("start_multitaper")
    downsampled_data_points = downsample_EGG(data_points)
    multitaper_df = apply_multitaper(downsampled_data_points)
    timer.print_duration_since("start_multitaper", "Time for multitaper")
    timer.set_time_point("start_smoothing")
    sxx_df = do_smoothing(multitaper_df, timer)
    sxx_norm = pd.DataFrame(data=sxx_df.T.values, columns=multitaper_df.columns, index=multitaper_df.index)
    timer.print_duration_since("start_smoothing", "Time to process spectrum")
    return sxx_norm


def downsample_EGG(eeg_data):
    '''
    Downsample the data to a target frequency

    You can also replace the Butterworth filter with Bessel filter or the default Chebyshev filter.
    system = dlti(*bessel(4,0.99))
    system = dlti(*cheby1(3,0.05,0.99))
    All filters produced very similar results for downsampling from 200Hz to 100Hz
    '''
    eeg_fs = Config.eeg_fs
    target_fs = Config.downsample_fs
    eeg_fs = round(eeg_fs)
    rate = eeg_fs / target_fs
    system = dlti(*butter(4, 0.99))
    return decimate(eeg_data, round(rate), ftype=system, zero_phase=True)


def apply_multitaper(data_points):
    eeg_fs = Config.eeg_fs
    eeg_data = np.array(data_points)
    start = [pd.to_datetime('today')]
    window_length = 4 * int(eeg_fs)
    window_step = 2 * int(eeg_fs)
    window_starts = np.arange(0, len(eeg_data) - window_length + 1, window_step)

    eeg_segs = detrend(eeg_data[list(map(lambda x: np.arange(x, x + window_length), window_starts))])

    freqs, psd_est, var_or_nu = tsa.multi_taper_psd(eeg_segs, Fs=eeg_fs, NW=4, adaptive=False, jackknife=False,
                                                    low_bias=True)

    time_idx = pd.date_range(start=start[0], freq='{}ms'.format(window_step / eeg_fs * 1000),
                             periods=len(psd_est))
    return pd.DataFrame(index=time_idx, data=psd_est, columns=freqs)


def apply_savgol_filter(x):
    return scipy.signal.savgol_filter(x, Config.savgol_window, Config.savgol_order)


def do_smoothing(multitaper_df, timer, iterations=Config.smoothing_iterations):
    timer.set_time_point("start_log_calc")
    # Log scale
    sxx_df = 10 * np.log(multitaper_df.T)
    timer.print_duration_since("start_log_calc")

    timer.set_time_point("start_savgol")
    # horizontal axis (time)
    for i in range(iterations):
        sxx_df = sxx_df.apply(apply_savgol_filter, axis=1, result_type='expand')

    timer.print_duration_since("start_savgol")
    return sxx_df


