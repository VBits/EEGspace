import Common
import Storage
import Config
from OnlineAnalysis import Config as OnlineConfig
from OfflineAnalysis import Config as OfflineConfig
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import pandas as pd

def create_required_files_for_online_mode(mouse_id):

    #original code for getting the required data

    # load up the data
    multitaper_data_path = OnlineConfig.multitaper_data_file_path.format(mouse_id=mouse_id)
    multitaper_df = Storage.load_from_file(multitaper_data_path)
    states_file_path = OnlineConfig.state_file_path.format(mouse_id=mouse_id)
    states = np.array(Storage.load_from_file(states_file_path))
    states_numeric = Common.states_to_numeric_version(states)

    # convert to medians
    # TODO MG change center to False, eliminate transformation below
    medians = multitaper_df.rolling(OnlineConfig.median_filter_buffer, center=True, win_type=None,
                                    min_periods=2).median()
    mid = OnlineConfig.median_filter_buffer_middle
    set_size = 200000

    # combine data
    combined_data = np.hstack((np.array(medians[:-mid]), np.array(multitaper_df[mid:])))
    series = combined_data[:set_size]
    states_numeric = states_numeric[mid:set_size + mid]
    Storage.dump_to_file(OnlineConfig.combined_data_file_path.format(mouse_id=mouse_id), combined_data)

    saved_state_averages = Storage.load_from_file(OfflineConfig.state_averages_path)
    data_state_averages = pd.DataFrame()
    for label in np.unique(states['clusters_knn']):
        data_state_averages[label] = multitaper_df.loc[states[states['clusters_knn'] == label].index].mean(axis=0)

    state_mappings = {}
    for data_state_average in data_state_averages:
        lowest_error = None
        state_mappings[data_state_average] = None
        for saved_state_average in saved_state_averages:
            error = sklearn.metrics.mean_squared_error(data_state_average, saved_state_average)
            if lowest_error is None or error < lowest_error:
                lowest_error = error
                state_mappings[data_state_average] = saved_state_average

    # load the state averages for the data that was used with the knn
    # get the averages of the states for the data that is loaded up
    # compare the two using the least error between the two?
    # find the label for the data average add it as the key in a dict
    # find the label of the matching knn data average add this as the value
    # when all is done add this into the state mapping file

    lda = Storage.load_from_file(OnlineConfig.lda_file_path.format(mouse_id=mouse_id))
    lda_transformed = lda.transform(series)
    # #create and save lda
    # lda = LDA(n_components=3)
    # lda_transformed = lda.fit_transform(series, states_numeric)
    # Storage.dump_to_file(OnlineConfig.lda_file_path.format(mouse_id=mouse_id), lda)
    #
    # create and save knn
    neigh = KNeighborsClassifier(n_neighbors=8)
    neigh.fit(lda_transformed, states_numeric)
    Storage.dump_to_file(OnlineConfig.knn_file_path.format(mouse_id=mouse_id), neigh)



    #-----------------------
    #new code for converting the states

    #load up the data
    multitaper_data_path = OnlineConfig.multitaper_data_file_path.format(mouse_id=mouse_id)
    multitaper_df = Storage.load_from_file(multitaper_data_path)
    states_file_path = OnlineConfig.state_file_path.format(mouse_id=mouse_id)
    states_df = Storage.load_from_file(states_file_path)
    states = np.array(states_df)
    states_numeric = Common.states_to_numeric_version(states)

    # # convert to medians
    # #TODO MG change center to False, eliminate transformation below
    medians = multitaper_df.rolling(OnlineConfig.median_filter_buffer, center=True, win_type=None, min_periods=2).median()
    # mid = OnlineConfig.median_filter_buffer_middle
    # set_size=200000
    #
    # #combine data
    # combined_data = np.hstack((np.array(medians[:-mid]),np.array(multitaper_df[mid:])))
    # series = combined_data[:set_size]
    # states_numeric = states_numeric[mid:set_size + mid]
    # Storage.dump_to_file(OnlineConfig.combined_data_file_path.format(mouse_id=mouse_id), combined_data)

    # combined_data = Storage.load_from_file(OnlineConfig.combined_data_file_path.format(mouse_id=mouse_id))
    # series = combined_data[:set_size]
    # states_numeric = states_numeric[mid:set_size + mid]

    combined_data = pd.concat([medians, multitaper_df], axis=1)

    #saved_state_averages = Storage.load_from_file(OfflineConfig.state_averages_path)
    data_state_averages = pd.DataFrame()
    for label in np.unique(states_df['states']):
        indexes = states_df[states_df['states'] == label].index
        data_state_averages[label] = combined_data.loc[indexes].mean(axis=0)


    # state_mappings = {}
    # for data_state_average in data_state_averages:
    #     lowest_error = None
    #     state_mappings[data_state_average] = None
    #     for saved_state_average in saved_state_averages:
    #         error = sklearn.metrics.mean_squared_error(data_state_average, saved_state_average)
    #         if lowest_error is None or error < lowest_error:
    #             lowest_error = error
    #             state_mappings[data_state_average] = saved_state_average


    #load the state averages for the data that was used with the knn
    #get the averages of the states for the data that is loaded up
    #compare the two using the least error between the two?
    #find the label for the data average add it as the key in a dict
    #find the label of the matching knn data average add this as the value
    #when all is done add this into the state mapping file

    lda = Storage.load_from_file(OnlineConfig.lda_file_path.format(mouse_id=mouse_id))
    #lda_transformed = lda.transform(series)
    # #create and save lda
    # lda = LDA(n_components=3)
    # lda_transformed = lda.fit_transform(series, states_numeric)
    # Storage.dump_to_file(OnlineConfig.lda_file_path.format(mouse_id=mouse_id), lda)
    #
    #create and save knn
    # neigh = KNeighborsClassifier(n_neighbors=8)
    # neigh.fit(lda_transformed, states_numeric)
    # Storage.dump_to_file(OnlineConfig.knn_file_path.format(mouse_id=mouse_id), neigh)
    knn = Storage.load_from_file(OnlineConfig.knn_file_path.format(mouse_id=mouse_id))
    knn_transformed = knn.predict(lda.transform(np.array(data_state_averages.transpose())))
    mapping = {data_state_averages.columns[i]: knn_transformed[i] for i in range(len(data_state_averages.columns)) }
    print(mapping)