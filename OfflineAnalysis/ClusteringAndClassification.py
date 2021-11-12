import OfflineAnalysis.ANN as ANN
from OfflineAnalysis.PlottingUtils import get_random_idx
from OfflineAnalysis.ProcessSmrx import get_mouse
from OfflineAnalysis.Config import *
import numpy as np
import pandas as pd
from Transformations import train_lda
from PlottingUtils import *
from NearestNeighbors import *
import joblib

m = get_mouse()

m.state_df = pd.read_pickle(EphysDir+Folder + 'states_210409_210409_B6Jv_m1.pkl')

######################################
# 1. Label the multitaper_df using an ANN (use 50 epochs, centered around the epoch of interest)
def chunker_list(seq, size):
    return [seq[i:i + size] for i in range(len(seq) - 1)]
def expand_epochs(m,smoothed_data=False,win=51):
    if smoothed_data:
        print ('Using savgol smoothed data')
        data_chunks = chunker_list(m.Sxx_df,win)
    else:
        print ('Using multitaper data')
        data_chunks = chunker_list(m.multitaper_df, win)
    #trim off the epochs at the end that don't have the same length
    data_chunks = data_chunks[:-win+1]
    data_chunks = np.asarray(data_chunks)
    #flatten the epochs into a single vector
    extended_data = data_chunks.reshape(data_chunks.shape[0],-1)
    #make epoch of interest the center one in the data
    extended_data = pd.DataFrame(data=extended_data, index=m.Sxx_df[win//2:-win+win//2].index)
    state_data = m.state_df[win//2:-win+win//2]
    return extended_data, state_data

# Sxx_extended, labels_extended = expand_epochs(m,smoothed_data=True)
Sxx_extended = pd.read_pickle(EphysDir + Folder + 'Sxx_extended_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], m.genotype, m.pos))
labels_extended = pd.read_pickle(EphysDir + Folder + 'labels_extended_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], m.genotype, m.pos))
#trim
m.state_df = m.state_df.iloc[:len(Sxx_extended)]


rand_idx = get_random_idx(Sxx_extended)


model = ANN.create_model(Sxx_extended)
# ANN.standardize_state_codes(m.state_df)
ANN.standardize_state_codes(labels_extended)
# ANN.plot_model(EphysDir+Folder)
classWeight = ANN.calculate_weights(m,rand_idx)
model = ANN.train_model(model,Sxx_extended,labels_extended,classWeight,rand_idx)


model.save_weights(EphysDir + Folder+ 'weights_Sxx_df_51epochs_centered_final.h5')
# model.load_weights(EphysDir+Folder,'weights_Sxx_df_51epochs_centered_final.h5')
labels = ANN.get_labels(model, m,Sxx_extended)
# NNetwork.test_accuracy()
#
#
#
# ######################################
# 2. Use ANN labels for LDA
rand_idx = get_random_idx(m.state_df)

#train an LDA based on the ANN labels
lda, X_train = train_lda(m.Sxx_df,m.state_df['states'],rand_idx,components=3)
# Create dataframe for LDs
m.LD_df = lda_transform_df(m,lda)

labels ='ann_states'
labels ='states'
title= 'LDA no labels', title= 'LDA original labels', title= 'LDA ann labels'
plot_LDA(m,rand_idx,states=True,labels=labels,savefigure=False)
plt.savefig(m.figureFolder+'LDA no labels' + m.figure_tail, dpi=dpi)


######################################
# 3 Density peak clustering
# build the density peak clusterer
clu = get_dpc(m,rand_idx,savefigure=True)
# decide the cutoffs for the clusters
clu = dpc_cutoffs(clu,100,2,savefigure=False)

# ######################################
# 4. Propagate DPC labels
clf = propagate_classes(m,rand_idx,clu,n_neighbors=201)


plt.savefig(m.figureFolder+'LDA dpc labels' + m.figure_tail, dpi=dpi)

### ----
# 3D
# Plot and evaluate state assignment 3D
plot_LDA(m,rand_idx,states=True,labels='states',savefigure=False)
plt.savefig(m.figureFolder + 'LDA DPC labels {}_{}'.format(Folder[:6], File[:6]) + m.figure_tail, dpi=dpi)


# Load OR Save knn model
knn_file = EphysDir+Folder + 'knn_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
# knn_file = EphysDir+Folder + 'knn_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,'all_mice')
### -------------------
# Save file
joblib.dump(clf, knn_file)
### -------------------
# Recover previously saved file
clf = joblib.load(knn_file)

### -------------------
#Save State Dataframe
m.state_df.to_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
# #Load previously saved Dataframe from experimental folder
m.state_df = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))

# ######################################
# # 5. Use DPC labels to update LDA

#train an LDA based on the ANN labels
lda, X_train = train_lda(m.Sxx_df,m.state_df['states'],rand_idx,components=3)
# Create dataframe for LDs
m.LD_df = lda_transform_df(m,lda)


labels ='states'
title= 'LDA dpc labels'
plot_LDA(m,rand_idx,states=True,labels=labels,savefigure=False)
plt.savefig(m.figureFolder+'LDA dpc labels updated LDA' + m.figure_tail, dpi=dpi)


# ### -------------------
# Store or load LDA transformation
lda_filename = EphysDir+Folder + 'lda_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
# Save file
joblib.dump(lda, lda_filename)
# # Recover previously saved file
lda = joblib.load(lda_filename)




