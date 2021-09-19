import pandas as pd

import Common
import Config
import Storage
from Common import load_offline_data, states_to_numeric_version, apply_savgol
from smrx_version.ANN import try_ann
import numpy as np
from Common import train_lda


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


def create_statespace_from_prev_savgol(series, states, buffer=50, subset_size=None, load_from_file=True,
                                       concatenate_original=False, use_ann=False):
    if subset_size is None:
        subset_size = np.array(series).shape[0]

    save_file_path = Config.base_path + "/prev_" + str(buffer) + "_epochs_" + str(
        subset_size) + "_long_savgol_smoothed.pkl"
    params = {'series': series, 'buffer': buffer, 'subset_size': subset_size}
    new_series = Storage.load_or_recreate_file(save_file_path, smooth_prev_epochs_with_savgol, not load_from_file,
                                               params)

    states_numeric = states_to_numeric_version(states)[buffer:subset_size]

    series = np.array(series[buffer:subset_size])

    if concatenate_original:
        concatenated = []
        for i in range(0, states_numeric.shape[0]):
            concatenated.append(np.concatenate((new_series[i], series[i]), axis=0))
        new_series = np.array(concatenated)

    if use_ann:
        state_predictions_numeric_original = try_ann(pd.DataFrame(series), states_numeric, False)

        percentage = Common.transition_state_misclassification_percentages(states_numeric,
                                                                           state_predictions_numeric_original)

        print(percentage)

        Common.print_confusion_matrix(states_numeric, state_predictions_numeric_original)

        state_predictions_numeric_prev_epochs = try_ann(pd.DataFrame(new_series), states_numeric, False)

        percentage = Common.transition_state_misclassification_percentages(states_numeric,
                                                                           state_predictions_numeric_prev_epochs)

        print(percentage)

        Common.print_confusion_matrix(states_numeric, state_predictions_numeric_prev_epochs)

        general_accuracy = Common.compare_states_for_accuracy(state_predictions_numeric_original,
                                                              state_predictions_numeric_prev_epochs)
        transition_accuracy = Common.transition_state_misclassification_percentages(state_predictions_numeric_original,
                                                                                    state_predictions_numeric_prev_epochs)

        print("General accuracy:" + str(general_accuracy) + "%")
        print("Transition accuracy:" + str(transition_accuracy) + "%")

    def train_lda_for_object_retrieval(series, numerical_states, show_plot):
        lda, lda_trained_data = train_lda(series, numerical_states, show_plot)
        return {'lda': lda, 'lda_trained_data': lda_trained_data}

    params = {'series': new_series[:100000], 'numerical_states': states_numeric[:100000], 'show_plot': True}

    lda_result_new = Storage.load_or_recreate_file(Config.base_path + "/lda_for_subset.pkl",
                                               train_lda_for_object_retrieval, True, params)

    # params = {'series': series[:100000], 'numerical_states': states_numeric[:100000], 'show_plot': True}
    # lda_result_original = Storage.load_or_recreate_file(Config.base_path + "/lda_for_subset.pkl",
    #                                                train_lda_for_object_retrieval, params)

    lda_data_new = lda_result_new['lda'].transform(new_series)
    # lda_data_original = lda_result_original['lda'].transform(series)

    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=8)
    neigh.fit(lda_data_new, states_numeric)
    knn_predictions_numeric = neigh.predict(lda_data_new)

    Common.print_confusion_matrix(states_numeric, knn_predictions_numeric)
    general_accuracy = Common.compare_states_for_accuracy(states_numeric, knn_predictions_numeric)
    transition_accuracy = Common.transition_state_misclassification_percentages(states_numeric, knn_predictions_numeric)

    print("General accuracy:" + str(general_accuracy) + "%")
    print("Transition accuracy:" + str(transition_accuracy) + "%")

def train_ann_on_transitions(series, states, buffer=50, subset_size=None, load_from_file=True):
    if subset_size is None:
        subset_size = np.array(series).shape[0]

    save_file_path = Config.base_path + "/prev_" + str(buffer) + "_epochs_" + str(
        subset_size) + "_long_savgol_smoothed.pkl"
    params = {'series': series, 'buffer': buffer, 'subset_size': subset_size}
    new_series = Storage.load_or_recreate_file(save_file_path, smooth_prev_epochs_with_savgol, not load_from_file,
                                               params)

    states_numeric = states_to_numeric_version(states)[buffer:subset_size]

    series = np.array(series[buffer:subset_size])

    #get the transitions
    transition_indexes = Common.get_transition_indexes(states_numeric)
    bulked_out = np.zeros(len(series))
    #bulked_out[transition_indexes[:, 2]] = (states_numeric[transition_indexes[:, 2]] + 1)
    bulked_out[transition_indexes[:, 2]] = 1

    states_numeric = bulked_out

    state_predictions_numeric_original = try_ann(pd.DataFrame(series), states_numeric, False)

    percentage = Common.transition_state_misclassification_percentages(states_numeric,
                                                                       state_predictions_numeric_original)

    print(percentage)

    Common.print_confusion_matrix(states_numeric, state_predictions_numeric_original)

    state_predictions_numeric_prev_epochs = try_ann(pd.DataFrame(new_series), states_numeric, False)

    percentage = Common.transition_state_misclassification_percentages(states_numeric,
                                                                       state_predictions_numeric_prev_epochs)

    print(percentage)

    Common.print_confusion_matrix(states_numeric, state_predictions_numeric_prev_epochs)

    general_accuracy = Common.compare_states_for_accuracy(state_predictions_numeric_original,
                                                          state_predictions_numeric_prev_epochs)
    transition_accuracy = Common.transition_state_misclassification_percentages(state_predictions_numeric_original,
                                                                                state_predictions_numeric_prev_epochs)

    print("General accuracy:" + str(general_accuracy) + "%")
    print("Transition accuracy:" + str(transition_accuracy) + "%")



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
    if False:
        Common.graph_transition_averages(unsmoothed, states)
    train_ann_on_transitions(unsmoothed, states)
    #create_statespace_from_prev_savgol(unsmoothed, states)
    # create_statespace_from_last_epoch_averages(unsmoothed, states)
