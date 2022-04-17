"""
v.2022 - Not finalized yet
"""
import OfflineAnalysis.ANN as ANN
from OfflineAnalysis.GeneralUtils import *
from OfflineAnalysis.LoadData import *
from OfflineAnalysis.Transformations import *
from OfflineAnalysis.PlottingUtils import *
from OfflineAnalysis.NearestNeighbors import *
from OfflineAnalysis import Config as OfflineConfig
from Pipeline import DPA
import joblib
from sklearn.metrics import confusion_matrix

######################################
def process_EEG_data():
    #1. Get data for indicated genotype and channel.
    # Preprocess data, or specify to load preprocessed data
    m = load_EEG_data(OfflineConfig.mouse_description, OfflineConfig.mouse_id)

    #2. Load previously saved files and average files
    # Recover previously saved LDA file
    load_LDA = query_yes_no("Would you like to load a previously saved LDA?")
    if load_LDA:
        lda = joblib.load(OfflineConfig.lda_filename)
    # Recover previously saved state dataframe
    load_state_df = query_yes_no("Would you like to load a previously saved state dataframe?")
    if load_state_df:
        m.state_df = pd.read_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + OfflineConfig.state_df_filename)
    # Recover previously saved KNN file
    load_KNN = query_yes_no("Do you want to load the previously stored KNN model?")
    if load_KNN:
        knn_clf = joblib.load(OfflineConfig.knn_file)


    #Create an extended dataframe that contains the smoothed and raw epochs
    m.Sxx_ext = expand_epochs(m)
    rand_idx = get_random_idx(m.Sxx_ext,size=OfflineConfig.random_epoch_size)

    ############################################################
    # 2b. Create the LDA space
    use_ann = query_yes_no("Do you want to get ANN labels and use them to train an LDA?")
    if use_ann:
        print('Using an ANN to get provisional labels and train a new LDA')
        ANN.label_data(m,reuse_weights=True)
        # Create LDA using the ANN labels (works better with noise free data)
        lda, x_train = train_lda(m.Sxx_ext, m.state_df['ann_labels'], rand_idx, components=OfflineConfig.LDA_components)

    else:
        print ('Using an LDA trained on multiple animals')
        #Load a previously created LDA (works better with noisy data)
        lda_filename = OfflineConfig.offline_data_path + 'lda_average.joblib'
        lda = joblib.load(lda_filename)

    # Reduce dimensionality of data and save in a new dataframe
    m.LD_df = lda_transform_df(m.Sxx_ext,lda)
    # Evaluate LDA space
    plot_LDA(m,rand_idx)

    ######################################
    # 4. Density peak clustering
    # Find density peaks in low dimensional space, tweak Z
    est = DPA.DensityPeakAdvanced(Z=OfflineConfig.dpa_z,k_max=OfflineConfig.dpa_k_max)
    est.fit(m.LD_df.loc[rand_idx])

    # Plot DPA clusters on LDA
    plot_DPA_LDA(m, rand_idx, est)

    repeat_dpa = query_yes_no("Do you want to repeat DPA with different settings?")
    previous_value = OfflineConfig.dpa_z
    while repeat_dpa:
        updated_dpa_z = input("Provide a new Z - number of standard deviations, previous value was {}".format(previous_value))
        previous_value = updated_dpa_z
        repeat_dpa = query_yes_no("Do you want to repeat DPA with different settings?")

        # remap the spurious clusters into 4 labels,
    label_dict = {}
    for label in range(4):
        # Taking multiple inputs
        label_dict[label] = {int(x.strip()) for x in input("Clusters to merge into label {}, use comma separated format eg. 2,1".format(label)).split(',')}

    label_dict = {vi: k for k, v in label_dict.items() for vi in v}


    est.labels_ = np.vectorize(label_dict.get)(est.labels_)

    # Decide if you want to update LDA using the DPA clusters (optional step)
    retrain_LDA = query_yes_no("Do you want to retrain LDA using the new DPA labels?")
    if retrain_LDA:
        lda, x_train = train_lda_dpa_labels(m.Sxx_ext,est,rand_idx,components=OfflineConfig.LDA_components)
        m.LD_df = lda_transform_df(m.Sxx_ext,lda)


    # ######################################
    # 5. Propagate DPC labels
    knn_clf = get_knn_clf(m,rand_idx,est,n_neighbors=OfflineConfig.KNN_n_neighbors)
    # propagate labels
    m.knn_pred(knn_clf, m.Sxx_ext,state_averages_path)

    # Plot and evaluate state assignment 3D
    plot_LDA(m,rand_idx,m.state_df['states'])

    #######################################
    # 6. Save files
    ### -------------------
    # Save State Dataframe
    save_state_df = query_yes_no("Do you want to store the state dataframe?")
    if save_state_df:
        m.state_df.to_pickle(base_path + experimental_path + state_df_filename)
    # Save LDA transformation
    save_LDA = query_yes_no("Do you want to store the LDA?")
    if save_LDA:
        joblib.dump(lda, lda_filename)
    # Save knn model
    save_KNN = query_yes_no("Do you want to store the KNN model for this mouse?")
    if save_KNN:
        joblib.dump(knn_clf, knn_file)

process_EEG_data()

#######################################
#-------------------------------------------------
#Optional. Confusion matrix and accuracy score
# #-------------------------------------------------
# compute_confusion_matrix = query_yes_no("Do you want to create a confusion matrix and an accuracy score? [y/n]")
# if compute_confusion_matrix:
#     m.state_df = pd.read_pickle(base_path + experimental_path + 'states_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
#     m.state_df_ground_truth = pd.read_pickle(base_path + experimental_path + 'states-corr_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
#     confusion_matrix(m.state_df_ground_truth['states'],m.state_df['states'],normalize='true',labels=["SWS", "REM", "HTwake","LTwake"])
#     accuracy_score(m.state_df_ground_truth['states'],m.state_df['states'])