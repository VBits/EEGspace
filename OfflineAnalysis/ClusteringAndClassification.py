"""
v.20211116 - Not finalized yet
"""
import OfflineAnalysis.ANN as ANN
from OfflineAnalysis.PlottingUtils import get_random_idx
from OfflineAnalysis.ProcessSmrx import get_mouse
from OfflineAnalysis.Config import *
# import numpy as np
import pandas as pd
from DensityPeaks import *
from Transformations import *
from PlottingUtils import *
from NearestNeighbors import *
import joblib
from Expand import *

m = get_mouse('SertCre-CS',1)

######################################
# 1. Label the multitaper_df using an ANN (use 50 epochs, centered around the epoch of interest)
#Create the Sxx_extended
Sxx_extended = expand_epochs(m,smoothed_data=True)
rand_idx = get_random_idx(Sxx_extended)

# #OR load previously calculated files
# Sxx_extended = pd.read_pickle(EphysDir + Folder + 'Sxx_extended_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], m.genotype, m.pos))
# labels_extended = pd.read_pickle(EphysDir + Folder + 'labels_extended_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], m.genotype, m.pos))

#Create and train a model
model = ANN.create_model(Sxx_extended)
ANN.standardize_state_codes(labels_extended)
ANN.plot_model(EphysDir+Folder)
classWeight = ANN.calculate_weights(m,rand_idx)
model = ANN.train_model(model,Sxx_extended,labels_extended,classWeight,rand_idx)
model.save_weights(EphysDir + Folder+ 'weights_Sxx_df_51epochs_centered_final.h5')

#load weights
model.load_weights(EphysDir+Folder+'weights_Sxx_df_51epochs_centered_final.h5')
#
labels = ANN.get_labels(model, m,Sxx_extended)
# NNetwork.test_accuracy()
#
############################################################
# ######################################
# 2. Use ANN labels for LDA
#train an LDA based on the ANN labels
lda, X_train = train_lda(Sxx_extended,m.state_df['states'],rand_idx,components=3)
#load LDA
lda_filename = EphysDir+Folder + 'lda_extended-corr_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
# Save file
joblib.dump(lda, lda_filename)
# # Recover previously saved file
lda = joblib.load(lda_filename)

# Create dataframe for LDs
m.LD_df = lda_transform_df(Sxx_extended,lda)

labels ='ann_states'
title= 'LDA ann labels'
plot_LDA(m,rand_idx,states=False,labels=labels,savefigure=False)
plt.savefig(m.figureFolder+title + m.figure_tail, dpi=dpi)

######################################
# 3 Density peak clustering
# build the density peak clusterer
clu = get_dpc(m,rand_idx,savefigure=True)
# decide the cutoffs for the clusters
clu = dpc_cutoffs(clu,50,2,savefigure=False)

# ######################################
# 4. Propagate DPC labels
clf = propagate_classes(m,Sxx_extended,rand_idx,state_averages_path,clu,n_neighbors=201)

### ----
# 3D
# Plot and evaluate state assignment 3D
labels ='states'
plot_LDA(m,rand_idx,states=True,labels=labels,savefigure=False)
plt.savefig(m.figureFolder + 'LDA DPC labels {}_{}'.format(Folder[:6], File[:6]) + m.figure_tail, dpi=dpi)


# Load OR Save knn model
knn_file = EphysDir+Folder + 'knn_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
### -------------------
# Save file
joblib.dump(clf, knn_file)
### -------------------
# Recover previously saved file
clf = joblib.load(knn_file)

### -------------------
#Save State Dataframe
m.state_df.to_pickle(EphysDir + Folder + 'states-corr_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
# #Load previously saved Dataframe from experimental folder
m.state_df = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))

# ######################################
# # 5. Optional: Use DPC labels to update LDA

lda, X_train = train_lda(Sxx_extended,m.state_df['states'],rand_idx,components=3)
# Create dataframe for LDs
m.LD_df = lda_transform_df(Sxx_extended,lda)

labels ='states'
plot_LDA(m,rand_idx,states=True,labels=labels,savefigure=False)
plt.savefig(m.figureFolder+'LDA dpc labels updated LDA' + m.figure_tail, dpi=dpi)

# ### -------------------
# Store or load LDA transformation
lda_filename = EphysDir+Folder + 'lda_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
# Save file
joblib.dump(lda, lda_filename)
# # Recover previously saved file
lda = joblib.load(lda_filename)




