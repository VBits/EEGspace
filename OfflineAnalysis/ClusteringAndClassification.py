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
from SVM import *


# m = get_mouse('SertCre-CS',1)
for i in range(1, 9):
# for i in range(9, 17):
    print (i)
    m = get_mouse('SertCre-CS',i,load=False)

m = get_mouse('B6J',1,load=True)
######################################
# 1. Label the multitaper_df using an ANN (use 50 epochs, centered around the epoch of interest)
#Create the Sxx_extended
# Sxx_extended = expand_epochs(m,smoothed_data=True,win=81)
rand_idx = get_random_idx(Sxx_extended,size=80000)


############################################################
# # 2a. Train SVM
# svm_clf = create_svm(m.Sxx_df3,m.state_df_corr,return_score=True,save=False)
# svm_filename = offline_data_path + 'svm_B6J_m1.joblib'
# joblib.dump(svm_clf, svm_filename)

############################################################
# 2a. Train ANN
NNetwork = ANN(multitaper_extended,labels_extended,rand_idx)
NNetwork.initialize()
NNetwork.train_model()
NNetwork.get_labels()
NNetwork.save_weights(EphysDir + Folder, 'weights_Multitaper_df_51epochs_centered.h5')
NNetwork.load_weights(EphysDir+Folder,'weights_Sxx_df.h5')
NNetwork.test_accuracy()

############################################################
# 2b. Get temporary SVM labels
# Recover previously saved model
svm_clf = load_svm(offline_data_path)
m.svm_labels = get_svm_labels(m.Sxx_df3,svm_clf)
m.svm_labels = get_svm_labels(m.multitaper_df,svm_clf)

############################################################
# 3. Train an LDA or load a previously created LDA
# Recover previously saved model
#train an LDA based on the SVM labels
lda, X_train = train_lda(Sxx_extended,m.svm_labels['svm_labels'],rand_idx,components=3)
lda, X_train = train_lda(Sxx_extended,m.state_df['states'],rand_idx,components=3)
lda, X_train = train_lda(Sxx_extended.loc['2021-04-09 13:55:04':'2021-04-13 14:01:54'],m.state_df_corr['states'],rand_idx,components=3)
lda, X_train = train_lda(m.Sxx_df3,m.state_df_corr['states'],rand_idx,components=3)
lda, X_train = train_lda(m.Sxx_df3,m.svm_labels['svm_labels'],rand_idx,components=3)
lda, X_train
# Create dataframe for LDs
m.LD_df = lda_transform_df(Sxx_extended,lda)
m.LD_df = lda_transform_df(m.Sxx_df3,lda)

#TODO test PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
Sxx_extended_scaled =MinMaxScaler().fit_transform(Sxx_extended)
pca = PCA(n_components=3)
LD_df = pca.fit_transform(Sxx_extended_scaled)
LD_df = pca.fit_transform(Sxx_extended)
m.LD_df = pd.DataFrame(data=LD_df, columns=['LD1', 'LD2', 'LD3'], index=Sxx_extended.index)
m.Sxx_df_scaled =MinMaxScaler().fit_transform(m.Sxx_df)
LD_df = pca.fit_transform(m.Sxx_df3)
m.LD_df = pd.DataFrame(data=LD_df, columns=['LD1', 'LD2', 'LD3'], index=m.Sxx_df.index)


#TODO Import NMF
from sklearn.decomposition import NMF
# Create an NMF instance: model
model = NMF(n_components=3)
# Fit the model to televote_Rank
m.Sxx_df_scaled =MinMaxScaler().fit_transform(Sxx_extended)
model.fit(m.Sxx_df_scaled)
# Transform the televote_Rank: nmf_features
LD = model.transform(m.Sxx_df_scaled)
m.LD_df = pd.DataFrame(data=LD, columns=['LD1', 'LD2', 'LD3'], index=m.Sxx_df.index)
# Print the NMF features
print(nmf_features.shape)
print(model.components_.shape)




#load previously saved LDA
lda_filename = offline_data_path +'lda_extended_211014_211102_Vglut2Cre-CS_m9.joblib'
lda_filename = EphysDir+Folder  + 'lda_extended-corr_210409_210409_B6J_m1.joblib'
lda_filename = EphysDir+Folder + 'lda_extended_210409_210409_B6J_m1.joblib'
lda_filename = EphysDir+Folder + 'lda_extended_21win_p4_210409_210409_B6J_m1.joblib'
lda = joblib.load(lda_filename)
# Create dataframe for LDs
m.LD_df = lda_transform_df(Sxx_extended,lda)
#Evaluate SVM labels
plot_LDA(m,rand_idx,savefigure=False)
#Or evaluate LDA shape seperately
title= 'LDA no labels', title= 'LDA svm labels'
plot_LDA(m,rand_idx,savefigure=False)
plt.savefig(m.figureFolder+title + m.figure_tail, dpi=dpi)

######################################
# 4. Density peak clustering
# build the density peak clusterer
clu = get_dpc(m,rand_idx,savefigure=False)
# decide the cutoffs for the clusters
clu = dpc_cutoffs(clu,100,4,savefigure=False)

# ######################################
# 5. Propagate DPC labels
knn_clf = get_knn_clf(m,rand_idx,clu,n_neighbors=201)
# propagate labels
m.knn_pred(knn_clf, Sxx_extended,state_averages_path)

##################
# #Automatic state assignment
# Nclusters = len(self.state_df['clusters_knn'].unique())
#
# state_averages = pd.read_pickle(state_averages_path)
# label_averages = pd.DataFrame()
# for label in np.unique(self.state_df['clusters_knn']):
#     label_averages[label] = self.Sxx_df_2.loc[self.state_df[self.state_df['clusters_knn'] == label].index].mean(axis=0)
# for label in ['HTwake','LTwake','SWS','REM']:
#     label_averages[label] = Sxx_combined_2.loc[states_combined[states_combined['states'] == label].index].mean(axis=0)
# plt.figure()
# plt.plot(label_averages)
# plt.legend()
#
# label_averages.to_pickle(EphysDir+Folder+'StateAverages.pkl')
# stateAverages = label_averages
#
# normalization = m.Sxx_df.quantile(q=0.01,axis=0)
# m.Sxx_df_norm = m.Sxx_df - normalization
#
# background = Sxx_combined.quantile(q=0.01,axis=0)
# Sxx_combined_2  = Sxx_combined - background
#
#
# plt.figure()
# plt.plot(state_averages)
# plt.legend()
# label_averages.plot()
#
# from scipy.signal import correlate
#
# for i in range(4):
#     print (i, stateAverages.corr(label_averages))
#
# label_averages.corrwith(stateAverages['SWS'])
#
# state_averages.corrwith(label_averages)
# pd.concat([state_averages, label_averages], axis=1).corr().iloc[:4,5:]
#
# if Nclusters == 4:
#     state_dict = {}
#     for state in state_averages:
#         print(state)
#         state_correlations = label_averages.corrwith(stateAverages[state])
#         state_dict[state] = state_correlations.argmax()
#
#     self.state_df['states'] = self.state_df['clusters_knn']
#     self.state_df.replace({"states": state_dict}, inplace=True)
# else:
#     print('Number of clusters not recognized. Automatic state assignment failed')
### ----
# 3D
# Plot and evaluate state assignment 3D
labels ='states'
labels = 'clusters_knn'
plot_LDA(m,rand_idx,m.state_df_corr['states'],savefigure=False)
plt.savefig(m.figureFolder + 'LDA DPC labels {}_{}'.format(Folder[:6], File[:6]) + m.figure_tail, dpi=dpi)


### -------------------
# Load OR Save knn model
knn_file = EphysDir+Folder + 'knn_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
# Save file
joblib.dump(clf, knn_file)
# Recover previously saved file
clf = joblib.load(knn_file)

### -------------------
#Save State Dataframe
m.state_df.to_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
# #Load previously saved Dataframe from experimental folder
m.state_df_corr = pd.read_pickle(EphysDir + Folder + 'states-corr_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))

# ######################################
# # 5. Use DPC labels to update LDA

#train an LDA based on the ANN labels
lda, X_train = train_lda(Sxx_extended,m.state_df_corr['states'],rand_idx,components=3)
# Create dataframe for LDs
m.LD_df = lda_transform_df(Sxx_extended,lda)


labels ='states'
title= 'LDA dpc labels'
plot_LDA(m,rand_idx,m.state_df['states'],savefigure=False)
plot_LDA(m,rand_idx,m.state_df_corr['states'],savefigure=False)
plot_LDA(m,rand_idx,m.svm_labels['svm_labels'])
plt.savefig(m.figureFolder+'LDA corr labels multitaper data another rand_idx' + m.figure_tail, dpi=dpi)

# ### -------------------
# Store or load LDA transformation
lda_filename = EphysDir+Folder + 'lda_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
# Save file
joblib.dump(lda, lda_filename)
# # Recover previously saved file
lda = joblib.load(lda_filename)

#-------------------------------------------------
from sklearn.metrics import confusion_matrix
m.state_df = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
m.state_df_corr = pd.read_pickle(EphysDir + Folder + 'states-corr_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
confusion_matrix(m.state_df_corr['states'],m.state_df['states'],normalize='true')

