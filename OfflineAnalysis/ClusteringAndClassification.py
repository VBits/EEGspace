"""
v.20211116 - Not finalized yet
"""
import OfflineAnalysis.ANN as ANN
from OfflineAnalysis.GeneralUtils import get_random_idx
from OfflineAnalysis.ProcessSmrx import get_mouse
from OfflineAnalysis.Config import *
# import numpy as np
import pandas as pd
from GeneralUtils import *
from DensityPeaks import *
from Transformations import *
from PlottingUtils import *
from NearestNeighbors import *
import joblib
from Expand import *
# from SVM import *
from Pipeline import DPA
import time

######################################
#1. Get data for indicated genotype and channel
m = get_mouse('Vglut2Cre-CS',13,load=True)

#Create an extended dataframe that contains the smoothed and raw epochs
m.Sxx_ext = expand_epochs(m)
rand_idx = get_random_idx(m.Sxx_ext,size=20000)

############################################################
# 2a. Train ANN
model = ANN.create_model(m.Sxx_ext)
ANN.standardize_state_codes(m.state_df)
# ANN.plot_model(EphysDir+Folder)
classWeight = ANN.calculate_weights(m,rand_idx)
model = ANN.train_model(model,m.Sxx_ext,m.state_df,classWeight,rand_idx)


model.save_weights(offline_data_path+ 'weights_Sxx_df_sert_m1_medianfiltered.h5')

############################################################
# 2b. Get temporary labels from ANN
model.load_weights(offline_data_path + 'weights_Sxx_df_sert_m1_medianfiltered.h5')
m.state_df = pd.DataFrame(index=m.Sxx_ext.index)
m.state_df['ann_labels'] = ANN.get_labels(model,m.Sxx_ext)


############################################################
# 3a. Train an LDA on temporary ANN labels
#train an LDA based on the ANN labels
lda, X_train = train_lda(m.Sxx_ext,m.state_df['ann_labels'],rand_idx,components=3)
# Create dataframe for LDs
m.LD_df = lda_transform_df(m.Sxx_ext,lda)

############################################################
# 3b. Load a previously created LDA
lda_filename = offline_data_path +'lda_211014_211102_Vglut2Cre-CS_m9.joblib'
lda_filename = EphysDir+Folder + 'lda_210409_210409_B6J_m1.joblib'
lda = joblib.load(lda_filename)
# Create dataframe for LDs
m.LD_df = lda_transform_df(m.Sxx_ext,lda)
#Evaluate LDA transformation labels
title= 'LDA no labels'
plot_LDA(m,rand_idx,savefigure=False)
plt.savefig(m.figureFolder+title + m.figure_tail, dpi=dpi)

######################################
# 4. Density peak clustering
# # Find density peaks in low dimensional space
est = DPA.DensityPeakAdvanced(Z=0.90,k_max=201)
start=time.time()
est.fit(m.LD_df.loc[rand_idx])
end=time.time()
print(end-start)

#plot DPA clusters on LDA
def plot_DPA_LDA(m, rand_idx, dpa=est)

#Train new LDA based on DPA clusters
lda, X_train = train_lda_dpa_labels(m.Sxx_ext,est,rand_idx,components=3)
m.LD_df = lda_transform_df(m.Sxx_ext,lda)


# ######################################
# 5. Propagate DPC labels
knn_clf = get_knn_clf(m,rand_idx,est,n_neighbors=201)
# propagate labels
m.knn_pred(knn_clf, m.Sxx_ext,state_averages_path)

########
# Plot and evaluate state assignment 3D
plot_LDA(m,rand_idx,m.state_df['states'],savefigure=False)
plt.savefig(m.figureFolder + 'LDA DPC labels {}_{}'.format(Folder[:6], File[:6]) + m.figure_tail, dpi=dpi)


### -------------------
# Load OR Save knn model
knn_file = EphysDir+Folder + 'knn_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
# Save file
joblib.dump(clf, knn_file)
# Recover previously saved file
clf = joblib.load(knn_file)


#######################################
#5. Check DPA labels
plot_LDA(m,rand_idx,m.state_df['states'],savefigure=False)
plt.savefig(m.figureFolder+'LDA corr labels multitaper data another rand_idx' + m.figure_tail, dpi=dpi)
### -------------------
#Save State Dataframe
m.state_df.to_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
# #Load previously saved Dataframe from experimental folder
m.state_df = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
# ### -------------------
# Store or load LDA transformation
lda_filename = EphysDir+Folder + 'lda_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
# Save file
joblib.dump(lda, lda_filename)
# # Recover previously saved file
lda = joblib.load(lda_filename)

#-------------------------------------------------
#Optional
#-------------------------------------------------
from sklearn.metrics import confusion_matrix
m.state_df = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
m.state_df_corr = pd.read_pickle(EphysDir + Folder + 'states-corr_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
confusion_matrix(m.state_df_corr['states'],m.state_df['states'],normalize='true')

#TODO test overlap
from sklearn.metrics import confusion_matrix,accuracy_score
x211 = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}_labelsDPA_Sxx201.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
x3 = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}_labelsDPA.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))

confusion_matrix(x3['states'],x211['states'],labels=["SWS", "REM", "HTwake","LTwake"])
accuracy_score(x3['states'],x211['states'])