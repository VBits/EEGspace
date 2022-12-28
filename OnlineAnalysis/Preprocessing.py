"""
Offline analysis
"""

import nitime.algorithms as tsa
from scipy.signal import detrend, dlti, butter, decimate
from OnlineAnalysis import Config
import numpy as np
import pandas as pd

def transform_data(data_points, timer):
    timer.set_time_point("start_multitaper")
    downsampled_data_points = downsample_EGG(data_points)
    multitaper_df = apply_multitaper(downsampled_data_points)
    timer.print_duration_since("start_multitaper", "Time for multitaper")
    medians = multitaper_df.rolling(Config.median_filter_buffer, center=True, win_type=None, min_periods=2).median()
    combined_data = np.hstack((np.array(medians)[-Config.median_filter_buffer_middle], np.array(multitaper_df)[-1]))
    return combined_data

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
    eeg_fs = Config.downsample_fs
    eeg_data = np.array(data_points)
    start = [pd.to_datetime('today')]
    window_length = 4 * int(eeg_fs)
    window_step = 2 * int(eeg_fs)
    window_starts = np.arange(0, len(eeg_data) - window_length + 1, window_step)
    eeg_dat_to_detrend = []
    for indexes in list(map(lambda x: np.arange(x, x + window_length), window_starts)):
        eeg_dat_to_detrend.append(eeg_data[indexes])
    eeg_segs = detrend(eeg_dat_to_detrend)

    freqs, psd_est, var_or_nu = tsa.multi_taper_psd(eeg_segs, Fs=eeg_fs, NW=4, adaptive=False, jackknife=False,
                                                    low_bias=True)

    time_idx = pd.date_range(start=start[0], freq='{}ms'.format(window_step / eeg_fs * 1000),
                             periods=len(psd_est))
    multitaper_df = pd.DataFrame(index=time_idx, data=psd_est, columns=freqs)
    return 10 * np.log(multitaper_df)




