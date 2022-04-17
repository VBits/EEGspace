"""
v.2022 - Not finalized yet
"""
import OfflineAnalysis.ANN as ANN
from OfflineAnalysis.GeneralUtils import *
from OfflineAnalysis.ProcessSmrx import *
from OfflineAnalysis.Transformations import *
from OfflineAnalysis.PlottingUtils import *
from OfflineAnalysis.NearestNeighbors import *
from OfflineAnalysis import Config as OfflineConfig
from Pipeline import DPA
import joblib
from sklearn.metrics import confusion_matrix

######################################
#1. Get data for indicated genotype and channel.
# Preprocess data, or specify to load preprocessed data
m = get_mouse(genotype,gene_pos,load=load_previously_analyzed_data)

#Create an extended dataframe that contains the smoothed and raw epochs
m.Sxx_ext = expand_epochs(m)
OfflineConfig.rand_idx = get_random_idx(m.Sxx_ext,size=random_epoch_size)

############################################################
# 2. Create the LDA space
if OfflineConfig.use_ANN:
    ANN.label_data(m,reuse_weights=True)
    # Create LDA using the ANN labels (works better with noise free data)
    lda, X_train = train_lda(m.Sxx_ext, m.state_df['ann_labels'], OfflineConfig.rand_idx, components=OfflineConfig.LDA_components)

else:
    #Load a previously created LDA (works better with noisy data)
    lda_filename = OfflineConfig.offline_data_path +'lda_average.joblib'
    lda = joblib.load(lda_filename)

# Reduce dimensionality of data and save in a new dataframe
m.LD_df = lda_transform_df(m.Sxx_ext,lda)
# Evaluate LDA space
plot_LDA(m,OfflineConfig.rand_idx)

######################################
# 4. Density peak clustering
# Find density peaks in low dimensional space, tweak Z
est = DPA.DensityPeakAdvanced(Z=OfflineConfig.DPA_Z,k_max=OfflineConfig.DPA_k_max)
est.fit(m.LD_df.loc[OfflineConfig.rand_idx])

# Plot DPA clusters on LDA
plot_DPA_LDA(m, OfflineConfig.rand_idx, est)

repeat_DPA = query_yes_no("Do you want to repeat DPA with different settings? Please respond with yes or no")
#TODO repeat DPA with user input for a different Z
if repeat_DPA:



# remap the spurious clusters into 4 labels,
# the order is merged_labels:spurious cluster labels
#TODO read this from file?
label_dict = {0:{2,1},
            1:{0},
            2:{3,5},
            3:{4}}
label_dict = {vi: k for k, v in label_dict.items() for vi in v}

est.labels_ = np.vectorize(label_dict.get)(est.labels_)

# Decide if you want to update LDA using the DPA clusters (optional step)
retrain_LDA = query_yes_no("Do you want to retrain LDA using the new DPA labels? Please respond with yes or no")
if retrain_LDA:
    lda, X_train = train_lda_dpa_labels(m.Sxx_ext,est,OfflineConfig.rand_idx,components=OfflineConfig.LDA_components)
    m.LD_df = lda_transform_df(m.Sxx_ext,lda)


# ######################################
# 5. Propagate DPC labels
knn_clf = get_knn_clf(m,OfflineConfig.rand_idx,est,n_neighbors=OfflineConfig.KNN_n_neighbors)
# propagate labels
m.knn_pred(knn_clf, m.Sxx_ext,state_averages_path)

# Plot and evaluate state assignment 3D
plot_LDA(m,OfflineConfig.rand_idx,m.state_df['states'])

#######################################
# 6. Save files
### -------------------
# Save or Load State Dataframe
state_df_filename = 'states_{}_{}_{}_m{}.pkl'.format(OfflineConfig.ExpDir[:6], OfflineConfig.File[:6], m.genotype, m.pos)
save_state_df = query_yes_no("Do you want to store the state dataframe? Please respond with yes or no")
if save_state_df:
    m.state_df.to_pickle(BaseDir + ExpDir + state_df_filename)
#TODO #Load previously saved Dataframe from experimental folder
load_state_df = query_yes_no("Would you like to load a previously saved state dataframe? Please respond with yes or no")
if load_state_df:
    m.state_df = pd.read_pickle(BaseDir + ExpDir + state_df_filename)
### -------------------
# Store or load LDA transformation
lda_filename = BaseDir + ExpDir + 'lda_{}_{}_{}_m{}.joblib'.format(OfflineConfig.ExpDir[:6], OfflineConfig.File[:6], m.genotype, m.pos)
# Save file
save_LDA = query_yes_no("Do you want to store the LDA? Please respond with yes or no")
if save_LDA:
    joblib.dump(lda, lda_filename)
# # Recover previously saved file
load_LDA = query_yes_no("Would you like to load a previously saved LDA? Please respond with yes or no")
if load_LDA:
    lda = joblib.load(lda_filename)
### -------------------
# Load OR Save knn model
knn_filename = BaseDir + ExpDir + 'knn_{}_{}_{}_m{}.joblib'.format(OfflineConfig.ExpDir[:6], OfflineConfig.File[:6], m.genotype, m.pos)
# Save file
save_KNN = query_yes_no("Do you want to store the KNN model for this mouse? Please respond with yes or no")
if save_KNN:
    joblib.dump(knn_clf, knn_file)
# Recover previously saved file
load_KNN = query_yes_no("Do you want to load the previously stored KNN model? Please respond with yes or no")
if save_KNN:
    knn_clf = joblib.load(OfflineConfig.knn_file)


#######################################
#-------------------------------------------------
#Optional. Confusion matrix and accuracy score
#-------------------------------------------------
compute_confusion_matrix = query_yes_no("Do you want to create a confusion matrix and an accuracy score? Please respond with yes or no")
if compute_confusion_matrix:
    m.state_df = pd.read_pickle(BaseDir + ExpDir + 'states_{}_{}_{}_m{}.pkl'.format(ExpDir[:6], File[:6], m.genotype, m.pos))
    m.state_df_ground_truth = pd.read_pickle(BaseDir + ExpDir + 'states-corr_{}_{}_{}_m{}.pkl'.format(ExpDir[:6], File[:6], m.genotype, m.pos))
    confusion_matrix(m.state_df_ground_truth['states'],m.state_df['states'],normalize='true',labels=["SWS", "REM", "HTwake","LTwake"])
    accuracy_score(m.state_df_ground_truth['states'],m.state_df['states'])