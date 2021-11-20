import pandas as pd
import scipy
from scipy.signal import savgol_filter

import Common
import Config
import Storage
from Common import load_offline_data, states_to_numeric_version, apply_savgol
from smrx_version.ANN import try_ann
import numpy as np
from Common import train_lda
from sklearn.neighbors import KNeighborsClassifier


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


def get_prev_epochs(series, buffer, subset_size, with_smoothing):
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
        if with_smoothing:
            subseries = apply_savgol(subseries, 41, 4)
        new_series.append(subseries)

    new_series = np.array(new_series).flatten().reshape(subset_size - buffer, epoch_size * buffer)
    return new_series


def try_ann_with_prev_and_current_epoch_probabiliites(series, states, buffer=50, subset_size=None, load_from_file=True):

    if subset_size is None:
        subset_size = np.array(series).shape[0]

    save_file_path = Config.base_path + "/prev_" + str(buffer) + "_epochs_" + str(
        subset_size) + "_long_savgol_smoothed.pkl"
    params = {'series': series, 'buffer': buffer, 'subset_size': subset_size}
    new_series = Storage.load_or_recreate_file(save_file_path, smooth_prev_epochs_with_savgol, not load_from_file,
                                               params)

    new_series = np.array(new_series)

    states_numeric = states_to_numeric_version(states)[buffer:subset_size]

    series = np.array(series[buffer:subset_size])

    state_predictions_probability_original = try_ann(pd.DataFrame(series), states_numeric, False, True)

    state_predictions_probability_prev_epochs = try_ann(pd.DataFrame(new_series), states_numeric, False, True)

    new_probabilities = np.dstack((state_predictions_probability_original, state_predictions_probability_prev_epochs))

    new_probabilities = new_probabilities.reshape(new_probabilities.shape[0], new_probabilities.shape[1] * new_probabilities.shape[2])

    state_predictions_combined_probabilities = try_ann(pd.DataFrame(new_probabilities), states_numeric, False, False)

    percentage = Common.transition_state_misclassification_percentages(states_numeric, state_predictions_combined_probabilities)

    print(percentage)

    Common.print_confusion_matrix(states_numeric, state_predictions_combined_probabilities)

def mirror_transform(series):
    window_size = 41
    def chunker(seq, size):
        return (seq.iloc[pos:pos + size] for pos in range(0, len(seq) - 1))

    def SG_filter(x):
        return scipy.signal.savgol_filter(x, window_size, 2, mode='mirror')

    chunk_len = 201
    smooth_iter = 3
    buffer = []
    for n, chunk in enumerate(chunker(series, chunk_len)):
        Sxx_mirror = pd.concat([chunk[:-1], chunk.iloc[::-1]], axis=0)
        for i in range(smooth_iter):
            Sxx_mirror = Sxx_mirror.apply(SG_filter, axis=0, result_type='broadcast')
        Sxx_savgol_current = Sxx_mirror.iloc[chunk_len - 1]
        buffer.append(Sxx_savgol_current)
    buffer = np.asarray(buffer)
    return buffer

def mirror_transform_faster(series):

    chunk_len = 41
    smooth_iter = 3
    window_size = 41

    series = np.array(series)
    series = [np.concatenate((series[pos:pos + chunk_len], series[pos:pos + chunk_len][::-1]), axis=0) for pos in range(len(series))]
    series = np.array(series)

    buffer = []
    for n, mirrored_chunk in enumerate(series):
        for i in range(smooth_iter):
            mirrored_chunk = scipy.signal.savgol_filter(mirrored_chunk, window_size, 2, mode='mirror', axis=0)
        if len(mirrored_chunk) > chunk_len - 1:
            buffer.append(mirrored_chunk[chunk_len - 1])
        else:
            buffer.append(mirrored_chunk[-1])
    buffer = np.asarray(buffer)
    return buffer


def try_knn_with_prev_and_current_epoch_probabiliites(series, states, buffer=50, subset_size=None, load_from_file=True):
    if subset_size is None:
        subset_size = np.array(series).shape[0]

    x = Common.get_state_mapping_from_list(np.array(states))
    print(x)

    standard_lda = Storage.load_from_file(Config.base_path + '/smrx_version/lda_210216_210301_Vglut2Cre-SuM_all_mice.joblib')

    save_file_path = Config.base_path + "/mirror_transformed_" + str(subset_size) + "_long.pkl"
    params = {'series': series }
    mirror_transformed = Storage.load_or_recreate_file(save_file_path, mirror_transform_faster, not load_from_file, params)

    mirror_transformed = mirror_transformed[buffer:subset_size]

    save_file_path = Config.base_path + "/prev_" + str(buffer) + "_epochs_" + str(
        subset_size) + "_long_savgol_smoothed.pkl"
    params = {'series': series, 'buffer': buffer, 'subset_size': subset_size}
    prev_epochs = Storage.load_or_recreate_file(save_file_path, smooth_prev_epochs_with_savgol, not load_from_file,
                                               params)



    states_numeric = states_to_numeric_version(states)[buffer:subset_size]
    Storage.dump_to_file(Config.base_path + "/ground_truth_states.pkl", states_numeric)

    def train_lda_for_object_retrieval(series, numerical_states, show_plot):
        lda, lda_trained_data = train_lda(series, numerical_states, show_plot)
        return {'lda': lda, 'lda_trained_data': lda_trained_data}

    params = {'series': mirror_transformed[:100000], 'numerical_states': states_numeric[:100000], 'show_plot': True}

    lda_result_log10 = Storage.load_or_recreate_file(Config.base_path + "/lda_for_mirror_transformed.pkl",
                                                   train_lda_for_object_retrieval, False, params)

    params = {'series': prev_epochs[:100000], 'numerical_states': states_numeric[:100000], 'show_plot': True}

    lda_result_prev_epoch = Storage.load_or_recreate_file(Config.base_path + "/lda_for_prev_epochs.pkl",
                                                   train_lda_for_object_retrieval, False, params)

    # lda_transformed_mirror_transformed = lda_result_log10['lda'].transform(mirror_transformed)
    lda_transformed_mirror_transformed = standard_lda.transform(mirror_transformed)
    Common.plot_lda(lda_transformed_mirror_transformed, states_numeric)
    Storage.dump_to_file(Config.base_path + "/mirror_lda_transformed_data_pretrained_lda.pkl", lda_transformed_mirror_transformed)

    ground_truth_lda_transformed_data_pretrained_lda = standard_lda.transform(series[buffer:])
    Common.plot_lda(ground_truth_lda_transformed_data_pretrained_lda, states_numeric)
    Storage.dump_to_file(Config.base_path + "/ground_truth_lda_transformed_data_pretrained_lda.pkl",
                         ground_truth_lda_transformed_data_pretrained_lda)


    lda_transformed_prev_epoch = lda_result_prev_epoch['lda'].transform(prev_epochs)
    # lda_transformed_prev_epoch = standard_lda.transform(prev_epochs)
    Common.plot_lda(lda_transformed_prev_epoch, states_numeric)

    neigh_mirror_transformed = KNeighborsClassifier(n_neighbors=8)
    neigh_mirror_transformed.fit(lda_transformed_mirror_transformed, states_numeric)
    state_predictions_probability_mirror_transformed = neigh_mirror_transformed.predict_proba(lda_transformed_mirror_transformed)

    state_predictions_mirror_transformed = neigh_mirror_transformed.predict(lda_transformed_mirror_transformed)
    Storage.dump_to_file(Config.base_path + "/state_predictions_for_mirror_transformed.pkl", state_predictions_mirror_transformed)
    general_accuracy_mirrored = Common.compare_states_for_accuracy(states_numeric, state_predictions_mirror_transformed)

    print(general_accuracy_mirrored)

    neigh_prev_epochs = KNeighborsClassifier(n_neighbors=8)
    neigh_prev_epochs.fit(lda_transformed_prev_epoch, states_numeric)
    state_predictions_probability_prev_epochs = neigh_prev_epochs.predict_proba(lda_transformed_prev_epoch)

    state_predictions_prev_epochs = neigh_prev_epochs.predict(lda_transformed_prev_epoch)
    Storage.dump_to_file(Config.base_path + "/state_predictions_for_prev_epochs.pkl", state_predictions_prev_epochs)
    general_accuracy_prev_epochs = Common.compare_states_for_accuracy(states_numeric, state_predictions_prev_epochs)

    print(general_accuracy_prev_epochs)

    new_probabilities = np.dstack((state_predictions_probability_mirror_transformed, state_predictions_probability_prev_epochs))

    new_probabilities = new_probabilities.reshape(new_probabilities.shape[0], new_probabilities.shape[1] * new_probabilities.shape[2])

    # state_predictions_combined_probabilities = try_ann(pd.DataFrame(new_probabilities), states_numeric, False, False)

    # params = {'series': new_probabilities[:100000], 'numerical_states': states_numeric[:100000], 'show_plot': True}
    #
    # lda_for_combined_probabilities = Storage.load_or_recreate_file(Config.base_path + "/lda_for_combined_probabilites.pkl",
    #                                                train_lda_for_object_retrieval, True, params)
    #
    # lda_combined_probabilities_transformed = lda_for_combined_probabilities['lda'].transform(new_probabilities)
    #
    neigh_combined_probabilities = KNeighborsClassifier(n_neighbors=8)
    neigh_combined_probabilities.fit(new_probabilities, states_numeric)
    state_predictions_combined_probabilities = neigh_combined_probabilities.predict(new_probabilities)
    Storage.dump_to_file(Config.base_path + "/state_predictions_for_combined_probabilities.pkl", state_predictions_combined_probabilities)

    general_accuracy = Common.compare_states_for_accuracy(states_numeric, state_predictions_combined_probabilities)

    print(general_accuracy)

    percentage = Common.transition_state_misclassification_percentages(states_numeric, state_predictions_combined_probabilities)

    print(percentage)

    Common.print_confusion_matrix(states_numeric, state_predictions_combined_probabilities)

def train_lda_for_object_retrieval(series, numerical_states, show_plot):
    lda, lda_trained_data = train_lda(series, numerical_states, show_plot)
    return {'lda': lda, 'lda_trained_data': lda_trained_data}

def try_knn_with_prev_and_current_epoch_probabiliites_1(series, states, buffer=50, subset_size=None, load_from_file=True):
    _, series, _, states = load_offline_data()
    if subset_size is None:
        subset_size = np.array(series).shape[0]

    # x = Common.get_state_mapping_from_list(np.array(states))
    # print(x)

    random_index_file_path = Config.base_path + "/rand_idx.joblib"
    random_indexes = np.array(Storage.load_from_file(random_index_file_path))
    indexes_to_remove = np.where(random_indexes < 50)
    random_indexes = np.delete(random_indexes, indexes_to_remove)

    overfit_states_file_path = Config.base_path + "/states_centered_multitaper_df.joblib"
    overfit_states_numeric = np.array(Storage.load_from_file(overfit_states_file_path))
    overfit_states_numeric = np.delete(overfit_states_numeric, indexes_to_remove)

    save_file_path = Config.base_path + "/prev_" + str(buffer) + "_epochs_" + str(
        subset_size) + "_long_savgol_smoothed.pkl"
    params = {'series': series, 'buffer': buffer, 'subset_size': subset_size}
    prev_epochs = Storage.load_or_recreate_file(save_file_path, smooth_prev_epochs_with_savgol, not load_from_file,
                                                params)

    states_numeric = states_to_numeric_version(states)[buffer:subset_size]

    # lda, lda_trained_data = train_lda(prev_epochs, states_numeric, True)

    lda_result_prev_epoch = Storage.load_or_recreate_file(Config.base_path + "/lda_for_prev_epochs.pkl",
                                                   train_lda_for_object_retrieval, False, params)
    lda_trained_data = lda_result_prev_epoch["lda"].transform(prev_epochs)

    # lda_trained_data = lda_result_prev_epoch["lda_trained_data"]

    Common.plot_lda(lda_trained_data[np.array(random_indexes) - 50], overfit_states_numeric)

    print("stop")
    # standard_lda = Storage.load_from_file(Config.base_path + '/smrx_version/lda_210216_210301_Vglut2Cre-SuM_all_mice.joblib')

    # save_file_path = Config.base_path + "/mirror_transformed_" + str(subset_size) + "_long.pkl"
    # params = {'series': series }
    # mirror_transformed = Storage.load_or_recreate_file(save_file_path, mirror_transform_faster, not load_from_file, params)
    #
    # mirror_transformed = mirror_transformed[buffer:subset_size]


    # Storage.dump_to_file(Config.base_path + "/ground_truth_states.pkl", states_numeric)

    # def train_lda_for_object_retrieval(series, numerical_states, show_plot):
    #     lda, lda_trained_data = train_lda(series, numerical_states, show_plot)
    #     return {'lda': lda, 'lda_trained_data': lda_trained_data}

    # params = {'series': mirror_transformed[:100000], 'numerical_states': states_numeric[:100000], 'show_plot': True}
    #
    # lda_result_log10 = Storage.load_or_recreate_file(Config.base_path + "/lda_for_mirror_transformed.pkl",
    #                                                train_lda_for_object_retrieval, False, params)

    # params = {'series': prev_epochs[:100000], 'numerical_states': states_numeric[:100000], 'show_plot': True}
    #
    # lda_result_prev_epoch = Storage.load_or_recreate_file(Config.base_path + "/lda_for_prev_epochs.pkl",
    #                                                train_lda_for_object_retrieval, False, params)



    # lda_transformed_mirror_transformed = lda_result_log10['lda'].transform(mirror_transformed)
    # lda_transformed_mirror_transformed = standard_lda.transform(mirror_transformed)
    # Common.plot_lda(lda_transformed_mirror_transformed, states_numeric)
    # Storage.dump_to_file(Config.base_path + "/mirror_lda_transformed_data_pretrained_lda.pkl",
    #                      lda_transformed_mirror_transformed)
    #
    # ground_truth_lda_transformed_data_pretrained_lda = standard_lda.transform(series[buffer:])
    # Common.plot_lda(ground_truth_lda_transformed_data_pretrained_lda, states_numeric)
    # Storage.dump_to_file(Config.base_path + "/ground_truth_lda_transformed_data_pretrained_lda.pkl",
    #                      ground_truth_lda_transformed_data_pretrained_lda)



    # lda_transformed_prev_epoch = lda_result_prev_epoch['lda'].transform(prev_epochs)
    # lda_transformed_prev_epoch = standard_lda.transform(prev_epochs)


    # from sklearn.neighbors import KNeighborsClassifier
    # neigh_mirror_transformed = KNeighborsClassifier(n_neighbors=8)
    # neigh_mirror_transformed.fit(lda_transformed_mirror_transformed, states_numeric)
    # state_predictions_probability_mirror_transformed = neigh_mirror_transformed.predict_proba(lda_transformed_mirror_transformed)
    #
    # state_predictions_mirror_transformed = neigh_mirror_transformed.predict(lda_transformed_mirror_transformed)
    # Storage.dump_to_file(Config.base_path + "/state_predictions_for_mirror_transformed.pkl", state_predictions_mirror_transformed)
    # general_accuracy_mirrored = Common.compare_states_for_accuracy(states_numeric, state_predictions_mirror_transformed)
    #
    # print(general_accuracy_mirrored)
    #
    # neigh_prev_epochs = KNeighborsClassifier(n_neighbors=8)
    # neigh_prev_epochs.fit(lda_transformed_prev_epoch, states_numeric)
    # state_predictions_probability_prev_epochs = neigh_prev_epochs.predict_proba(lda_transformed_prev_epoch)
    #
    # state_predictions_prev_epochs = neigh_prev_epochs.predict(lda_transformed_prev_epoch)
    # Storage.dump_to_file(Config.base_path + "/state_predictions_for_prev_epochs.pkl", state_predictions_prev_epochs)
    # general_accuracy_prev_epochs = Common.compare_states_for_accuracy(states_numeric, state_predictions_prev_epochs)
    #
    # print(general_accuracy_prev_epochs)
    #
    # new_probabilities = np.dstack((state_predictions_probability_mirror_transformed, state_predictions_probability_prev_epochs))
    #
    # new_probabilities = new_probabilities.reshape(new_probabilities.shape[0], new_probabilities.shape[1] * new_probabilities.shape[2])
    #
    # # state_predictions_combined_probabilities = try_ann(pd.DataFrame(new_probabilities), states_numeric, False, False)
    #
    # # params = {'series': new_probabilities[:100000], 'numerical_states': states_numeric[:100000], 'show_plot': True}
    # #
    # # lda_for_combined_probabilities = Storage.load_or_recreate_file(Config.base_path + "/lda_for_combined_probabilites.pkl",
    # #                                                train_lda_for_object_retrieval, True, params)
    # #
    # # lda_combined_probabilities_transformed = lda_for_combined_probabilities['lda'].transform(new_probabilities)
    # #
    # neigh_combined_probabilities = KNeighborsClassifier(n_neighbors=8)
    # neigh_combined_probabilities.fit(new_probabilities, states_numeric)
    # state_predictions_combined_probabilities = neigh_combined_probabilities.predict(new_probabilities)
    # Storage.dump_to_file(Config.base_path + "/state_predictions_for_combined_probabilities.pkl", state_predictions_combined_probabilities)
    #
    # general_accuracy = Common.compare_states_for_accuracy(states_numeric, state_predictions_combined_probabilities)
    #
    # print(general_accuracy)
    #
    # percentage = Common.transition_state_misclassification_percentages(states_numeric, state_predictions_combined_probabilities)
    #
    # print(percentage)
    #
    # Common.print_confusion_matrix(states_numeric, state_predictions_combined_probabilities)

def try_knn_with_prev_and_current_epoch_probabiliites(series, states, buffer=50, subset_size=None, load_from_file=True):
    series, states = load_offline_data()
    if subset_size is None:
        subset_size = np.array(series).shape[0]

    # x = Common.get_state_mapping_from_list(np.array(states))
    # print(x)

    random_index_file_path = Config.base_path + "/rand_idx.joblib"
    random_indexes = np.array(Storage.load_from_file(random_index_file_path))
    indexes_to_remove = np.where(random_indexes < 50)
    random_indexes = np.delete(random_indexes, indexes_to_remove)

    overfit_states_file_path = Config.base_path + "/states_centered_multitaper_df.joblib"
    overfit_states_numeric = np.array(Storage.load_from_file(overfit_states_file_path))
    overfit_states_numeric = np.delete(overfit_states_numeric, indexes_to_remove)

    save_file_path = Config.base_path + "/prev_" + str(buffer) + "_epochs_" + str(
        subset_size) + "_long_savgol_smoothed.pkl"
    params = {'series': series, 'buffer': buffer, 'subset_size': subset_size}
    prev_epochs = Storage.load_or_recreate_file(save_file_path, smooth_prev_epochs_with_savgol, not load_from_file,
                                                params)

    states_numeric = states_to_numeric_version(states)[buffer:subset_size]

    # lda, lda_trained_data = train_lda(prev_epochs, states_numeric, True)

    def train_lda_for_object_retrieval(series, numerical_states, show_plot):
        lda, lda_trained_data = train_lda(series, numerical_states, show_plot)
        return {'lda': lda, 'lda_trained_data': lda_trained_data}

    lda_result_prev_epoch = Storage.load_or_recreate_file(Config.base_path + "/lda_for_prev_epochs.pkl",
                                                   train_lda_for_object_retrieval, False, params)
    lda_trained_data = lda_result_prev_epoch["lda"].transform(prev_epochs)

    # lda_trained_data = lda_result_prev_epoch["lda_trained_data"]

    Common.plot_lda(lda_trained_data[np.array(random_indexes) - 50], overfit_states_numeric)

    print("stop")
    # standard_lda = Storage.load_from_file(Config.base_path + '/smrx_version/lda_210216_210301_Vglut2Cre-SuM_all_mice.joblib')

    # save_file_path = Config.base_path + "/mirror_transformed_" + str(subset_size) + "_long.pkl"
    # params = {'series': series }
    # mirror_transformed = Storage.load_or_recreate_file(save_file_path, mirror_transform_faster, not load_from_file, params)
    #
    # mirror_transformed = mirror_transformed[buffer:subset_size]


    # Storage.dump_to_file(Config.base_path + "/ground_truth_states.pkl", states_numeric)

    # def train_lda_for_object_retrieval(series, numerical_states, show_plot):
    #     lda, lda_trained_data = train_lda(series, numerical_states, show_plot)
    #     return {'lda': lda, 'lda_trained_data': lda_trained_data}

    # params = {'series': mirror_transformed[:100000], 'numerical_states': states_numeric[:100000], 'show_plot': True}
    #
    # lda_result_log10 = Storage.load_or_recreate_file(Config.base_path + "/lda_for_mirror_transformed.pkl",
    #                                                train_lda_for_object_retrieval, False, params)

    # params = {'series': prev_epochs[:100000], 'numerical_states': states_numeric[:100000], 'show_plot': True}
    #
    # lda_result_prev_epoch = Storage.load_or_recreate_file(Config.base_path + "/lda_for_prev_epochs.pkl",
    #                                                train_lda_for_object_retrieval, False, params)



    # lda_transformed_mirror_transformed = lda_result_log10['lda'].transform(mirror_transformed)
    # lda_transformed_mirror_transformed = standard_lda.transform(mirror_transformed)
    # Common.plot_lda(lda_transformed_mirror_transformed, states_numeric)
    # Storage.dump_to_file(Config.base_path + "/mirror_lda_transformed_data_pretrained_lda.pkl",
    #                      lda_transformed_mirror_transformed)
    #
    # ground_truth_lda_transformed_data_pretrained_lda = standard_lda.transform(series[buffer:])
    # Common.plot_lda(ground_truth_lda_transformed_data_pretrained_lda, states_numeric)
    # Storage.dump_to_file(Config.base_path + "/ground_truth_lda_transformed_data_pretrained_lda.pkl",
    #                      ground_truth_lda_transformed_data_pretrained_lda)



    # lda_transformed_prev_epoch = lda_result_prev_epoch['lda'].transform(prev_epochs)
    # lda_transformed_prev_epoch = standard_lda.transform(prev_epochs)


    # from sklearn.neighbors import KNeighborsClassifier
    # neigh_mirror_transformed = KNeighborsClassifier(n_neighbors=8)
    # neigh_mirror_transformed.fit(lda_transformed_mirror_transformed, states_numeric)
    # state_predictions_probability_mirror_transformed = neigh_mirror_transformed.predict_proba(lda_transformed_mirror_transformed)
    #
    # state_predictions_mirror_transformed = neigh_mirror_transformed.predict(lda_transformed_mirror_transformed)
    # Storage.dump_to_file(Config.base_path + "/state_predictions_for_mirror_transformed.pkl", state_predictions_mirror_transformed)
    # general_accuracy_mirrored = Common.compare_states_for_accuracy(states_numeric, state_predictions_mirror_transformed)
    #
    # print(general_accuracy_mirrored)
    #
    # neigh_prev_epochs = KNeighborsClassifier(n_neighbors=8)
    # neigh_prev_epochs.fit(lda_transformed_prev_epoch, states_numeric)
    # state_predictions_probability_prev_epochs = neigh_prev_epochs.predict_proba(lda_transformed_prev_epoch)
    #
    # state_predictions_prev_epochs = neigh_prev_epochs.predict(lda_transformed_prev_epoch)
    # Storage.dump_to_file(Config.base_path + "/state_predictions_for_prev_epochs.pkl", state_predictions_prev_epochs)
    # general_accuracy_prev_epochs = Common.compare_states_for_accuracy(states_numeric, state_predictions_prev_epochs)
    #
    # print(general_accuracy_prev_epochs)
    #
    # new_probabilities = np.dstack((state_predictions_probability_mirror_transformed, state_predictions_probability_prev_epochs))
    #
    # new_probabilities = new_probabilities.reshape(new_probabilities.shape[0], new_probabilities.shape[1] * new_probabilities.shape[2])
    #
    # # state_predictions_combined_probabilities = try_ann(pd.DataFrame(new_probabilities), states_numeric, False, False)
    #
    # # params = {'series': new_probabilities[:100000], 'numerical_states': states_numeric[:100000], 'show_plot': True}
    # #
    # # lda_for_combined_probabilities = Storage.load_or_recreate_file(Config.base_path + "/lda_for_combined_probabilites.pkl",
    # #                                                train_lda_for_object_retrieval, True, params)
    # #
    # # lda_combined_probabilities_transformed = lda_for_combined_probabilities['lda'].transform(new_probabilities)
    # #
    # neigh_combined_probabilities = KNeighborsClassifier(n_neighbors=8)
    # neigh_combined_probabilities.fit(new_probabilities, states_numeric)
    # state_predictions_combined_probabilities = neigh_combined_probabilities.predict(new_probabilities)
    # Storage.dump_to_file(Config.base_path + "/state_predictions_for_combined_probabilities.pkl", state_predictions_combined_probabilities)
    #
    # general_accuracy = Common.compare_states_for_accuracy(states_numeric, state_predictions_combined_probabilities)
    #
    # print(general_accuracy)
    #
    # percentage = Common.transition_state_misclassification_percentages(states_numeric, state_predictions_combined_probabilities)
    #
    # print(percentage)
    #
    # Common.print_confusion_matrix(states_numeric, state_predictions_combined_probabilities)

def create_statespace_from_prev_savgol(series, states, buffer=50, subset_size=None, load_from_file=True,
                                       concatenate_original=False, use_ann=True):
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
        transition_accuracy = Common.transition_state_misclassification_percentages(
            state_predictions_numeric_original,
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

# def generate_plots_for_blog(series):
#     #generate some savgol data with various window sizes and plot them
#     #shorter data
#     window_size = 29
#     poly_order = 1
#     subset = series[0:10]#Common.select_random_data(series, 15)
#     subset = np.array(subset).flatten()
#     subset = np.array([subset])
#     new_data = savgol_filter(subset, window_size, poly_order)
#     to_plot = np.stack([subset, new_data], axis=1)
#     labels = ["original data", "savgol smoothed"]
#     title = "Window size=" + str(window_size) + ", Polynomial order=" + str(poly_order)
#     colors = ['-b', '-g']
#     weights = [1,3]
#     Common.plot_data_overlayed(to_plot, title, labels, colors, weights)
#     print("done")


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


def train_lda_for_object_retrieval(series, numerical_states, show_plot):
    lda, lda_trained_data = train_lda(series, numerical_states, show_plot)
    return {'lda': lda, 'lda_trained_data': lda_trained_data}


def compare_confusion_matrices_multiple_data_shapes(series, buffer=50, subset_size=None, load_from_file=True):

    recreate_lda_file = False

    if subset_size is None:
        subset_size = np.array(series).shape[0]

    #todo for each case we are testing we need to train a new lda and new knn and compare the outputs of the confusion matrices and see the accuracy

    #get the overfit classes
    overfit_states_file_path = Config.base_path + "/states-corr_210409_210409_B6J_m1.pkl"
    overfit_states_numeric = Common.states_to_numeric_version(np.array(Storage.load_from_file(overfit_states_file_path)))
    overfit_states_numeric_prev_epochs = overfit_states_numeric[buffer:]

    #*get the prev epochs savgol*
    #-----------------------------------------------------------------------------------------------------------------
    #load the data
    save_file_path = Config.base_path + "/prev_" + str(buffer) + "_epochs_" + str(
        subset_size) + "_long_savgol_smoothed.pkl"
    params = {'series': series, 'buffer': buffer, 'subset_size': subset_size}
    prev_epochs_smoothed = Storage.load_or_recreate_file(save_file_path, smooth_prev_epochs_with_savgol, not load_from_file,
                                                params)[25:-26]

    #load or create the lda
    params = {'series': prev_epochs_smoothed[:100000], 'numerical_states': overfit_states_numeric_prev_epochs[:100000],
              'show_plot': True}
    lda_result_prev_epoch = Storage.load_or_recreate_file(Config.base_path + "/lda_for_prev_epochs.pkl",
                                                          train_lda_for_object_retrieval, recreate_lda_file, params)
    lda_transformed_prev_epoch = lda_result_prev_epoch["lda"].transform(prev_epochs_smoothed)



    #create the knn and apply it
    neigh_prev_epochs = KNeighborsClassifier(n_neighbors=8)
    neigh_prev_epochs.fit(lda_transformed_prev_epoch, overfit_states_numeric_prev_epochs)
    lda_states_numeric_prev_epoch = neigh_prev_epochs.predict(lda_transformed_prev_epoch)

    #get the confusion matrix
    Common.print_confusion_matrix(overfit_states_numeric_prev_epochs, lda_states_numeric_prev_epoch)
    print(Common.compare_states_for_accuracy(overfit_states_numeric_prev_epochs, lda_states_numeric_prev_epoch))

    #*get the log10 prev 50 epochs*
    #-----------------------------------------------------------------------------------------------------------------
    #load the data
    save_file_path = Config.base_path + "/prev_" + str(buffer) + "_epochs_" + str(
        subset_size) + "_long_not_smoothed.pkl"
    params = {'series': series, 'buffer': buffer, 'subset_size': subset_size, 'with_smoothing': False}
    prev_epochs_log10 = Storage.load_or_recreate_file(save_file_path, get_prev_epochs, not load_from_file, params)[25:-26]

    # load or create the lda
    params = {'series': prev_epochs_log10[:100000], 'numerical_states': overfit_states_numeric_prev_epochs[:100000], 'show_plot': True}
    lda_result_prev_epochs_log10 = Storage.load_or_recreate_file(Config.base_path + "/lda_for_prev_epochs_log10.pkl",
                                                          train_lda_for_object_retrieval, recreate_lda_file, params)
    lda_transformed_log10 = lda_result_prev_epochs_log10["lda"].transform(prev_epochs_log10)

    # create the knn and apply it
    neigh_prev_epoch_log10 = KNeighborsClassifier(n_neighbors=8)
    neigh_prev_epoch_log10.fit(lda_transformed_log10, overfit_states_numeric_prev_epochs)
    lda_states_numeric_prev_epoch_log10 = neigh_prev_epoch_log10.predict(lda_transformed_log10)

    # get the confusion matrix
    Common.print_confusion_matrix(overfit_states_numeric_prev_epochs, lda_states_numeric_prev_epoch_log10)
    print(Common.compare_states_for_accuracy(overfit_states_numeric_prev_epochs, lda_states_numeric_prev_epoch_log10))

    #*single log10 is the series itself*
    # -----------------------------------------------------------------------------------------------------------------
    log10 = np.array(series)[25:-26]

    # load or create the lda
    params = {'series': log10[:100000], 'numerical_states': overfit_states_numeric[:100000], 'show_plot': True}
    lda_result_log10 = Storage.load_or_recreate_file(Config.base_path + "/lda_for_log10.pkl",
                                                     train_lda_for_object_retrieval, recreate_lda_file, params)
    lda_transformed_log10 = lda_result_log10["lda"].transform(log10)

    # create the knn and apply it
    neigh_log10 = KNeighborsClassifier(n_neighbors=8)
    neigh_log10.fit(lda_transformed_log10, overfit_states_numeric)
    lda_states_numeric_log10 = neigh_log10.predict(lda_transformed_log10)

    # get the confusion matrix
    Common.print_confusion_matrix(overfit_states_numeric, lda_states_numeric_log10)
    print(Common.compare_states_for_accuracy(overfit_states_numeric, lda_states_numeric_log10))

    #Prev50 smoothed + prev 50 log 10
    # -----------------------------------------------------------------------------------------------------------------
    # prev_epochs_smoothed_and_prev_epochs_log_10 = np.hstack((prev_epochs_smoothed, prev_epochs_log10))
    #
    # # load or create the lda
    # params = {'series': prev_epochs_smoothed_and_prev_epochs_log_10[:100000],
    #           'numerical_states': overfit_states_numeric[:100000], 'show_plot': True}
    # lda_result_prev_epoch_smoothed_and_log10 = Storage.load_or_recreate_file(
    #     Config.base_path + "/lda_for_prev_epoch_smoothed_and_log10.pkl", train_lda_for_object_retrieval, False, params)
    # lda_transformed_prev_epochs_smoothed_and_log_10 = lda_result_prev_epoch_smoothed_and_log10["lda"]\
    #     .transform(prev_epochs_smoothed_and_prev_epochs_log_10)
    #
    # # create the knn and apply it
    # neigh_prev_epochs_smoothed_and_prev_epochs_log_10 = KNeighborsClassifier(n_neighbors=8)
    # neigh_prev_epochs_smoothed_and_prev_epochs_log_10.fit(lda_transformed_prev_epochs_smoothed_and_log_10, overfit_states_numeric)
    # lda_states_numeric_prev_epochs_smoothed_and_prev_epochs_log_10 = neigh_prev_epochs_smoothed_and_prev_epochs_log_10\
    #     .predict(prev_epochs_smoothed_and_prev_epochs_log_10)
    #
    # # get the confusion matrix
    # Common.print_confusion_matrix(overfit_states_numeric,
    #                               lda_states_numeric_prev_epochs_smoothed_and_prev_epochs_log_10)

    #Prev 50 smoothed + single log10 epoch
    # -----------------------------------------------------------------------------------------------------------------
    prev_epochs_smoothed_and_log_10 = np.hstack((prev_epochs_smoothed, log10[buffer:]))#todo check the dimensions

    # load or create the lda
    params = {'series': prev_epochs_smoothed_and_log_10[:50000], 'numerical_states': overfit_states_numeric_prev_epochs[:50000],
              'show_plot': True}
    lda_result_prev_epoch_smoothed_and_log10 = Storage.load_or_recreate_file(
        Config.base_path + "/lda_for_prev_epoch_smoothed_and_log10.pkl", train_lda_for_object_retrieval, recreate_lda_file, params)
    lda_transformed_prev_epochs_smoothed_and_log_10 = lda_result_prev_epoch_smoothed_and_log10["lda"] \
        .transform(prev_epochs_smoothed_and_log_10)

    # create the knn and apply it
    neigh_prev_epochs_smoothed_and_log_10 = KNeighborsClassifier(n_neighbors=8)
    neigh_prev_epochs_smoothed_and_log_10.fit(lda_transformed_prev_epochs_smoothed_and_log_10, overfit_states_numeric_prev_epochs)
    lda_states_numeric_prev_epochs_smoothed_and_log_10 = neigh_prev_epochs_smoothed_and_log_10 \
        .predict_(lda_transformed_prev_epochs_smoothed_and_log_10)

    # get the confusion matrix
    Common.print_confusion_matrix(overfit_states_numeric_prev_epochs, lda_states_numeric_prev_epochs_smoothed_and_log_10)
    print(Common.compare_states_for_accuracy(overfit_states_numeric_prev_epochs, lda_states_numeric_prev_epochs_smoothed_and_log_10))

    print("stop")

if __name__ == '__main__':
    multitaper, unsmoothed, smoothed, states = load_offline_data()
    if False:
        Common.graph_transition_averages(unsmoothed, states)
        train_ann_on_transitions(unsmoothed, states)
    #generate_plots_for_blog(unsmoothed)
    # try_ann_with_prev_and_current_epoch_probabiliites(unsmoothed, states)
    #try_knn_with_prev_and_current_epoch_probabiliites(unsmoothed, states)
    # create_statespace_from_prev_savgol(unsmoothed, states)
    # create_statespace_from_last_epoch_averages(unsmoothed, states)
    compare_confusion_matrices_multiple_data_shapes(unsmoothed)
