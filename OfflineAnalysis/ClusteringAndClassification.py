"""
v.20211116 - Not finalized yet
"""
import OfflineAnalysis.ANN as ANN
from OfflineAnalysis.GeneralUtils import *
from OfflineAnalysis.ProcessSmrx import *
from OfflineAnalysis.Config import *
from OfflineAnalysis.Expand import *

import pandas as pd
# from GeneralUtils import *
from DensityPeaks import *
from Transformations import *
from PlottingUtils import *
from NearestNeighbors import *
import joblib
from Pipeline import DPA

######################################
#1. Get data for indicated genotype and channel.
# Will preprocess data, unless you specify to load preprocessed data
m = get_mouse('cKO',10,load=True)

#Create an extended dataframe that contains the smoothed and raw epochs
m.Sxx_ext = expand_epochs(m)
rand_idx = get_random_idx(m.Sxx_ext,size=20000)

############################################################
# 2. Create ANN
model = ANN.create_model(m.Sxx_ext)

# a. Determine ANN weights
ANN.standardize_state_codes(m.state_df)
ANN.plot_model(model,EphysDir+Folder)
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

# a. Create LDA using the ANN labels
lda, X_train = train_lda(m.Sxx_ext,m.state_df['ann_labels'],rand_idx,components=3)

# b. Load a previously created LDA
lda_filename = offline_data_path +'lda_extended_211014_211102_Vglut2Cre-CS_m9.joblib'
lda = joblib.load(lda_filename)

# Create dataframe for LDs
m.LD_df = lda_transform_df(m.Sxx_ext,lda)
# Evaluate LDA space
plot_LDA(m,rand_idx,savefigure=False)
plt.savefig(m.figureFolder+ 'LDA no labels' + m.figure_tail, dpi=dpi)

######################################
# 4. Density peak clustering
# Find density peaks in low dimensional space
est = DPA.DensityPeakAdvanced(Z=1.2,k_max=201)
est.fit(m.LD_df.loc[rand_idx])

# Plot DPA clusters on LDA
plot_DPA_LDA(m, rand_idx, est)

# OPTIONAL Manually merge clusters
spurious_label = 3
correct_label = 0
est.labels_[est.labels_==spurious_label] = correct_label

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
plt.savefig(m.figureFolder + 'LDA DPC labels {}_{}'.format(Folder[:6], File[:6]) + m.figure_tail, dpi=dpi)

#######################################
# 6. Save files
### -------------------
# Save or Load State Dataframe
m.state_df.to_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
# #Load previously saved Dataframe from experimental folder
m.state_df = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
### -------------------
# Store or load LDA transformation
lda_filename = EphysDir+Folder + 'lda_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
# Save file
joblib.dump(lda, lda_filename)
# # Recover previously saved file
lda = joblib.load(lda_filename)
### -------------------
# Load OR Save knn model
knn_file = EphysDir+Folder + 'knn_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
# Save file
joblib.dump(clf, knn_file)
# Recover previously saved file
clf = joblib.load(knn_file)


#######################################
#-------------------------------------------------
#Optional. Confusion matrix and accuracy score
#-------------------------------------------------
from sklearn.metrics import confusion_matrix
m.state_df = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
m.state_df_ground_truth = pd.read_pickle(EphysDir + Folder + 'states-corr_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
confusion_matrix(m.state_df_ground_truth['states'],m.state_df['states'],normalize='true',labels=["SWS", "REM", "HTwake","LTwake"])
accuracy_score(m.state_df_ground_truth['states'],m.state_df['states'])