import pandas as pd
import Config
import Storage
from Common import load_offline_data, states_to_numeric_version, apply_savgol
from smrx_version.ANN import try_ann
import numpy as np
from Common import plot_transition_states, train_lda


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


def create_statespace_from_prev_savgol(series, states, buffer=50, subset_size=None, load_from_file=True):
    if subset_size is None:
        subset_size = np.array(series).shape[0]

    save_file_path = Config.base_path + "/prev_" + str(buffer) + "_epochs_" + str(subset_size) + "_long_savgol_smoothed.pkl"
    params = {'series': series, 'buffer': buffer, 'subset_size': subset_size}
    new_series = Storage.load_or_recreate_file(save_file_path, smooth_prev_epochs_with_savgol, not load_from_file,
                                               params)
    states_numeric = states_to_numeric_version(states)[buffer:subset_size]

    # train_lda(new_series, new_states_numeric, True)
    try_ann(pd.DataFrame(new_series), states_numeric, False)


def create_statespace_from_last_epoch_averages(series, states, number_of_averages=3):
    subset_size = 100000
    states_numeric = states_to_numeric_version(states)[:subset_size]
    buffer = number_of_averages
    new_series = []
    for i, epoch in enumerate(np.array(series[:subset_size])):
        if i < buffer:
            continue
        average = np.mean(np.array(series[i - buffer:i]), axis=0)
        new_series.append(average)

    new_series = np.array(new_series)
    new_states_numeric = states_numeric[buffer:]
    train_lda(new_series, new_states_numeric, True)


if __name__ == '__main__':
    multitaper, unsmoothed, smoothed, states = load_offline_data()
    create_statespace_from_prev_savgol(unsmoothed, states)
    # create_statespace_from_last_epoch_averages(unsmoothed, states)
