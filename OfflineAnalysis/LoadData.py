import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import pandas as pd
import sys
sys.path.append('C:/Users/bitsik0000/PycharmProjects/ClosedLoopEEG/OfflineAnalysis')
from OfflineAnalysis.Mouse import Mouse
from OfflineAnalysis import Config as OfflineConfig
from OfflineAnalysis.GeneralUtils import query_yes_no

def load_EEG_data(description, mouse_id):
    m = Mouse(description, mouse_id)

    # Create directory to save figures
    m.gen_folder(OfflineConfig.base_path, OfflineConfig.experimental_path)
    load_previously_analyzed_data = query_yes_no("Do you want to load previously analyzed data?")
    if load_previously_analyzed_data:
        print('Loading previously analysed file {}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
        # Load previously saved Dataframe from experimental folder
        m.Sxx_df = pd.read_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + 'Sxx_df_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
        m.multitaper_df = pd.read_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + 'Multitaper_df_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
    else:
        print('Processing EEG data and storing files: _{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
        #Load EEG data
        m.read_smrx(OfflineConfig.base_path, OfflineConfig.experimental_path, OfflineConfig.file)

        ### -------------------
        ### Downsample, perform multitaper and normalize data
        if m.EEG_fs > OfflineConfig.target_fs:
            print ('downsampling mouse {} EEG data, from {}Hz to {}Hz'.format(m.mouse_id,m.EEG_fs,OfflineConfig.target_fs))
            m.downsample_EGG(target_fs=OfflineConfig.target_fs)


        m.multitaper(resolution=OfflineConfig.epoch_seconds)
        m.smoothen_spectrum(window_size=OfflineConfig.smoothing_window)

        # Save normalized Dataframe to experimental folder
        m.Sxx_df.to_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + 'Sxx_df_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
        m.multitaper_df.to_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + 'Multitaper_df_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))

    return m