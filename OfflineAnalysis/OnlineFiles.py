import Common
import Storage
import Config
from OnlineAnalysis import Config as OnlineConfig
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np

def create_required_files_for_online_mode(mouse_num):

    # load up the data
    multitaper_data_path = OnlineConfig.multitaper_data_file_path.format(mouse_num=mouse_num)
    multitaper_df = Storage.load_from_file(multitaper_data_path)
    states_file_path = OnlineConfig.state_file_path.format(mouse_num=mouse_num)
    states = np.array(Storage.load_from_file(states_file_path))
    states_numeric = Common.states_to_numeric_version(states)

    # convert to medians
    #TODO MG change center to False, eliminate transformation bellow
    medians = multitaper_df.rolling(OnlineConfig.median_filter_buffer, center=True, win_type=None, min_periods=2).median()
    mid = OnlineConfig.median_filter_buffer_middle
    set_size=200000

    #combine data
    combined_data = np.hstack((np.array(medians[:-mid]),np.array(multitaper_df[mid:])))
    series = combined_data[:set_size]
    states_numeric = states_numeric[mid:set_size + mid]
    Storage.dump_to_file(OnlineConfig.combined_data_file_path.format(mouse_num=mouse_num), combined_data)

    lda = Storage.load_from_file(OnlineConfig.lda_file_path.format(mouse_num=mouse_num))
    lda_transformed = lda.transform(series)
    # #create and save lda
    # lda = LDA(n_components=3)
    # lda_transformed = lda.fit_transform(series, states_numeric)
    # Storage.dump_to_file(OnlineConfig.lda_file_path.format(mouse_num=mouse_num), lda)
    #
    #create and save knn
    neigh = KNeighborsClassifier(n_neighbors=8)
    neigh.fit(lda_transformed, states_numeric)
    Storage.dump_to_file(OnlineConfig.knn_file_path.format(mouse_num=mouse_num), neigh)
