"""
v.2022 - Not finalized yet
"""
import OfflineAnalysis.ANN as ANN
from OfflineAnalysis.GeneralUtils import *
from OfflineAnalysis.ProcessSmrx import *
from OfflineAnalysis.Transformations import *
from OfflineAnalysis.PlottingUtils import *
from OfflineAnalysis.NearestNeighbors import *
from Pipeline import DPA
import joblib

######################################
#1. Get data for indicated genotype and channel.
# Preprocess data, or specify to load preprocessed data
m = get_mouse('VgatCre_CS_YFP',13,load=True)

#Create an extended dataframe that contains the smoothed and raw epochs
m.Sxx_ext = expand_epochs(m)
rand_idx = get_random_idx(m.Sxx_ext,size=20000)

############################################################
# 2. Create ANN
model = ANN.create_model(m.Sxx_ext)

# a. Train ANN and save weights
ANN.standardize_state_codes(m.state_df)
ANN.plot_model(model, BaseDir + ExpDir)
classWeight = ANN.calculate_weights(m,rand_idx)
model = ANN.train_model(model,m.Sxx_ext,m.state_df,classWeight,rand_idx)
model.save_weights(offline_data_path+ 'weights_Sxx_ext.h5')

# b. Load previously determined ANN weights
model.load_weights(offline_data_path + 'weights_Sxx_ext.h5')

# Get labels from ANN
m.state_df = pd.DataFrame(index=m.Sxx_ext.index)
m.state_df['ann_labels'] = ANN.get_labels(model,m.Sxx_ext)

############################################################
# 3. Create the LDA space

# a. Create LDA using the ANN labels (works better with noise free data)
lda, X_train = train_lda(m.Sxx_ext,m.state_df['ann_labels'],rand_idx,components=3)

# b. Load a previously created LDA (works better with noisy data)
lda_filename = offline_data_path +'lda_average.joblib'
lda = joblib.load(lda_filename)

# Reduce dimensionality of data and save in a new dataframe
m.LD_df = lda_transform_df(m.Sxx_ext,lda)
# Evaluate LDA space
plot_LDA(m,rand_idx,savefigure=False)
plt.savefig(m.figureFolder+ 'LDA no labels' + m.figure_tail, dpi=dpi)

######################################
# 4. Density peak clustering
# Find density peaks in low dimensional space, tweak Z
est = DPA.DensityPeakAdvanced(Z=0.9,k_max=201)
est.fit(m.LD_df.loc[rand_idx])

# Plot DPA clusters on LDA
plot_DPA_LDA(m, rand_idx, est)


# OPTIONAL merge spurious clusters into 4 labels, labels:merged_labels
label_dict = {0:2,1:3,2:0,3:2,4:2,5:0,6:3,7:1}
est.labels_ = np.vectorize(label_dict.get)(est.labels_)

# OPTIONAL Update LDA using the DPA clusters
lda, X_train = train_lda_dpa_labels(m.Sxx_ext,est,rand_idx,components=3)
m.LD_df = lda_transform_df(m.Sxx_ext,lda)


# ######################################
# 5. Propagate DPC labels
knn_clf = get_knn_clf(m,rand_idx,est,n_neighbors=201)
# propagate labels
m.knn_pred(knn_clf, m.Sxx_ext,state_averages_path)

# Plot and evaluate state assignment 3D
plot_LDA(m,rand_idx,m.state_df['states'],savefigure=False)
plt.savefig(m.figureFolder + 'LDA DPC labels {}_{}'.format(ExpDir[:6], File[:6]) + m.figure_tail, dpi=dpi)

#######################################
# 6. Save files
### -------------------
# Save or Load State Dataframe
m.state_df.to_pickle(BaseDir + ExpDir + 'states_{}_{}_{}_m{}.pkl'.format(ExpDir[:6], File[:6], m.genotype, m.pos))
# #Load previously saved Dataframe from experimental folder
m.state_df = pd.read_pickle(BaseDir + ExpDir + 'states_{}_{}_{}_m{}.pkl'.format(ExpDir[:6], File[:6], m.genotype, m.pos))
### -------------------
# Store or load LDA transformation
lda_filename = BaseDir + ExpDir + 'lda_{}_{}_{}_m{}.joblib'.format(ExpDir[:6], File[:6], m.genotype, m.pos)
# Save file
joblib.dump(lda, lda_filename)
# # Recover previously saved file
lda = joblib.load(lda_filename)
### -------------------
# Load OR Save knn model
knn_file = BaseDir + ExpDir + 'knn_{}_{}_{}_m{}.joblib'.format(ExpDir[:6], File[:6], m.genotype, m.pos)
# Save file
joblib.dump(knn_clf, knn_file)
# Recover previously saved file
knn_file = offline_data_path +'knn_average.joblib'
knn_clf = joblib.load(knn_file)


#######################################
#-------------------------------------------------
#Optional. Confusion matrix and accuracy score
#-------------------------------------------------
from sklearn.metrics import confusion_matrix
m.state_df = pd.read_pickle(BaseDir + ExpDir + 'states_{}_{}_{}_m{}.pkl'.format(ExpDir[:6], File[:6], m.genotype, m.pos))
m.state_df_ground_truth = pd.read_pickle(BaseDir + ExpDir + 'states-corr_{}_{}_{}_m{}.pkl'.format(ExpDir[:6], File[:6], m.genotype, m.pos))
confusion_matrix(m.state_df_ground_truth['states'],m.state_df['states'],normalize='true',labels=["SWS", "REM", "HTwake","LTwake"])
accuracy_score(m.state_df_ground_truth['states'],m.state_df['states'])