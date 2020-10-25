import nitime.algorithms as tsa
from scipy.signal import detrend
import Config
import numpy as np
import pandas as pd
import scipy


def transform_data(data_points):
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
    multitaper_df = pd.DataFrame(index=time_idx, data=psd_est, columns=freqs)

    _, _, sxx_norm = process_spectrum(multitaper_df)
    return sxx_norm


def process_spectrum(multitaper_df):
    ## Normalize the data and plot density spectrogram
    def SG_filter(x):
        return scipy.signal.savgol_filter(x, 41, 2)

    # Log scale
    sxx_df = 10 * np.log(multitaper_df.T)

    # horizontal axis (time)
    iterations = Config.smoothing_iterations
    for i in range(iterations):
        sxx_df = sxx_df.apply(SG_filter, axis=1, result_type='expand')

    def density_calc(dataframe, boundary=(-100, 90)):
        # now calculate the bins for each frequency
        density_mat = []
        mean_density = []
        for i in range(len(dataframe.index)):
            density, bins = np.histogram(dataframe.iloc[i, :], bins=5000, range=boundary, density=True)
            density_mat.append(density)
            mean_density.append(dataframe.iloc[i, :].mean())
        density_mat = np.array(density_mat)
        bins = (bins[1:] + bins[:-1]) / 2
        return density_mat, bins

    density_mat, bins = density_calc(sxx_df, boundary=(-100, 90))  # -1,1550

    density_df = pd.DataFrame(index=bins, data=density_mat.T, columns=multitaper_df.columns)
    for i in range(iterations):
        density_df = density_df.apply(SG_filter, axis=0, result_type='expand')

    baseline = np.argmax(density_df.values > 0.01, axis=0)

    norm = 0 - bins[baseline]
    sxx_norm = sxx_df.add(norm, axis=0)
    density_norm, power_bins = density_calc(sxx_norm, boundary=(-25, 50))
    sxx_norm = pd.DataFrame(data=sxx_norm.T.values,columns=multitaper_df.columns, index=multitaper_df.index)

    return density_norm, power_bins, sxx_norm


