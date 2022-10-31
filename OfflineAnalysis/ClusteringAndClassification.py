import OfflineAnalysis.Utilities.PlottingUtils
from OfflineAnalysis.Utilities.GeneralUtils import *
from OfflineAnalysis.Utilities.LoadData import *
from OfflineAnalysis.Utilities.PlottingUtils import *
from OfflineAnalysis.Utilities.NearestNeighbors import *
from OfflineAnalysis.Utilities.Outliers import *
from OfflineAnalysis.Utilities.Clustering import *
import OfflineAnalysis.Config as OfflineConfig
import joblib
from sklearn.metrics import confusion_matrix,accuracy_score
import multiprocessing

"""
Run small chunks of code as needed and follow instructions within terminal
"""
if __name__ == '__main__':
    ########################################################################################################
    # 1. Get data for indicated genotype and channel.
    # Preprocess data, or load previously processed data
    m = process_EEG_data(OfflineConfig.mouse_description, OfflineConfig.mouse_id)
    rand_idx = get_random_idx(m.Sxx_ext, size=OfflineConfig.random_epoch_size)

    ########################################################################################################





    # 2. Create or load LDA space
    lda = get_LDA(m,rand_idx)
    # Evaluate LDA s1pace
    manager = multiprocessing.Manager()
    input_queue = manager.Queue()
    p = multiprocessing.Process(target=OfflineAnalysis.Utilities.PlottingUtils.plot_LDA_multiprocessing_version,
                                args=(m.LD_df.copy(), rand_idx, m.figureFolder, m.figure_tail, m.colors, input_queue))
    p.daemon = True
    p.start()

        # savefigure = query_yes_no("Do you want to save plot? Please respond with yes or no")
        # if savefigure:
        #     plt.savefig(m.figureFolder + figure_title + m.figure_tail,dpi=OfflineConfig.dpi)

    # proc = Process(target=plot_LDA, args=(m,rand_idx,))
    #         # procs.append(proc)
    # proc.start()

    #Optional: save figure
    #BugWarning: do not run together with plotting function
    savefigure_function_multiprocessing_version(input_queue)

    ########################################################################################################
    # 3. Optional: Load additional data if this mouse has been analyzed previously and then move to step 5
    # Recover previously saved state dataframe
    load_state_df = query_yes_no("Do you want to load a previously created: \nm.state_df?")
    if load_state_df:
        m.state_df = pd.read_pickle(OfflineConfig.state_df_filename)
    # Recover previously saved KNN file
    load_KNN = query_yes_no("Do you want to load a previously created: \nKNN model")
    if load_KNN:
        knn_clf = joblib.load(OfflineConfig.knn_filename)

    ########################################################################################################
    # 4. Density peak clustering
    # Find density peaks in low dimensional space,
    est = clustering_DPA_multiprocessing(m,rand_idx, input_queue, dpa_z=OfflineConfig.dpa_z)

    #Optional: Repeat clustering with DPA by tweaking Z, number of standard deviations
    #Hint: some times it can also help to select a new rand_idx
    # est = clustering_DPA(m,rand_idx,dpa_z=0.6)

    #Optional: remap the spurious clusters into 4 labels
    label_dict = {0: [1,4,3,6],
                  1: [0],
                  2: [2],
                  3: [5]}
    merging_spurious_labels_multiprocessing(m,rand_idx,label_dict,est, input_queue)

    # Optional: save figure
    # BugWarning: do not run together with plotting function
    savefigure_function_multiprocessing_version(input_queue)

    #Optional: Do you want to update LDA using the DPA clusters? (only when outliers not present)
    repeat_LDA = query_yes_no("Do you want to retrain LDA using the new DPA labels?")
    if repeat_LDA:
            lda = retrain_LDA(m,rand_idx,est)

    ########################################################################################################
    # 5. Propagate DPC labels
    knn_clf = get_knn_clf(m,rand_idx,est,n_neighbors=OfflineConfig.knn_n_neighbors)

    # propagate labels and evaluate assignment
    propagate_knn_labels_multiprocessing(m,rand_idx,knn_clf, input_queue)

    # Optional: save figure
    # BugWarning: do not run together with plotting function
    savefigure_function_multiprocessing_version(input_queue)

    ########################################################################################################
    # 6. Detect and label outliers (if any)

    # Train a density based model that will detect outliers (change EPS value as necessary)
    outlier_model = outlier_detection_multiprocessing(m,rand_idx,input_queue, eps=OfflineConfig.eps,min_samples = OfflineConfig.min_samples)
    # outlier_model = outlier_detection(m,rand_idx,eps=1.8)

    # Detect outliers in the whole recording
    predict_all_outliers_multiprocessing(m,rand_idx,outlier_model, queue, ambiguous_state=0)

    # Optional: save figure
    # BugWarning: do not run together with plotting function
    savefigure_function_multiprocessing_version(input_queue)

    ########################################################################################################
    # 7. Save files
    ### -------------------
    # Save State Dataframe
    save_state_df = query_yes_no("Do you want to store the state dataframe?")
    if save_state_df:
        m.state_df.to_pickle(OfflineConfig.state_df_filename)
    # Save LDA transformation
    save_LDA = query_yes_no("Do you want to store the LDA?")
    if save_LDA:
        joblib.dump(lda, OfflineConfig.lda_filename)
    # Save knn model
    save_KNN = query_yes_no("Do you want to store the KNN model for this mouse?")
    if save_KNN:
        joblib.dump(knn_clf, OfflineConfig.knn_filename)

    ########################################################################################################
    ########################################################################################################
    # -------------------------------------------------
    # Optional validation of results: Confusion matrix and accuracy score
    # -------------------------------------------------
    compute_confusion_matrix = query_yes_no("Do you want to create a confusion matrix and an accuracy score? [y/n]")
    if compute_confusion_matrix:
        #m.state_df was generated earlier. Ground truth data (eg. expert annotation) needs to be provided separately
        m.state_df_ground_truth = pd.read_pickle(OfflineConfig.base_path + OfflineConfig.experimental_path + 'states-corr_{}_{}_{}_m{}.pkl'.format(OfflineConfig.experiment_id, OfflineConfig.file_id, m.description, m.mouse_id))
        confusion_matrix(m.state_df_ground_truth['states'],m.state_df['states'],normalize='true',labels=["SWS", "REM", "HTwake","LTwake"])
        accuracy_score(m.state_df_ground_truth['states'],m.state_df['states'])