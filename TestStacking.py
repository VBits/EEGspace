import pandas as pd
from Common import load_offline_data, states_to_numeric_version, apply_savgol
from smrx_version.ANN import try_ann
import numpy as np
from Common import plot_transition_states, train_lda

def create_statespace_from_prev_savgol(series, states):
    subset_size = 100000
    states_numeric = states_to_numeric_version(states)[:subset_size]
    buffer = 7
    new_series = []
    for i, epoch in enumerate(np.array(series[:subset_size])):
        if i < buffer:
            continue
        subseries = np.array(series[i - buffer:i])
        subseries = [i2 for i1 in subseries for i2 in i1]
        subseries_smoothed = apply_savgol(subseries, 41, 4)
        new_series.append(subseries_smoothed)

    new_series = np.array(new_series)
    new_states_numeric = states_numeric[buffer:]
    train_lda(new_series, new_states_numeric, True)
    #try_ann(pd.DataFrame(new_series), new_states_numeric, False)


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
    #create_statespace_from_last_epoch_averages(unsmoothed, states)
