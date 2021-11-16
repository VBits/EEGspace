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
# ANN.standardize_state_codes(m.state_df)
ANN.standardize_state_codes(labels_extended)
# ANN.plot_model(EphysDir+Folder)
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
#Use SVM to gel labels
from sklearn.svm import LinearSVC
linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(Sxx_extended.iloc[rand_idx], m.state_df['states'].iloc[rand_idx])

Z = linear.predict(Sxx_extended.iloc[rand_idx])

svm_labels = pd.DataFrame(data=Z)


fig = plt.figure()
ax = Axes3D(fig)
# ax.scatter(m.LD_df['LD1'].iloc[rand_idx], m.LD_df['LD2'].iloc[rand_idx], m.LD_df['LD3'].iloc[rand_idx],
#            c=svm_labels[0].apply(lambda x: m.colors[x]), alpha=0.2,s=5,linewidths=0)
ax.scatter(m.LD_df['LD1'].loc[rand_idx], m.LD_df['LD2'].loc[rand_idx], m.LD_df['LD3'].loc[rand_idx],
           c=clu.membership,cmap='tab10',alpha=0.2,s=5,linewidths=0)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')

svm_filename = EphysDir+Folder + 'svm_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
svm_filename = EphysDir+Folder + 'svm_211011_211102_B6J_m1.joblib'
# Save file
joblib.dump(linear, svm_filename)
# # Recover previously saved file
linear = joblib.load(svm_filename)

#
# ######################################
# 2. Use ANN labels for LDA
#train an LDA based on the ANN labels
lda, X_train = train_lda(Sxx_extended,m.state_df['states'],rand_idx,components=3)
#load LDA
lda_filename = EphysDir+Folder + 'lda_extended-corr_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
# lda_filename = EphysDir+Folder + 'lda_extended_211014_211102_Vglut2Cre-CS_m9.joblib'
# lda_filename = EphysDir+Folder + 'lda_extended-corr_210409_210409_B6J_m1.joblib'
# Save file
joblib.dump(lda, lda_filename)
# # Recover previously saved file
lda = joblib.load(lda_filename)

# Create dataframe for LDs
m.LD_df = lda_transform_df(Sxx_extended,lda)

labels ='ann_states'
labels ='states'
title= 'LDA no labels', title= 'LDA dpc labels', title= 'LDA ann labels'
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
labels = 'clusters_knn'
plot_LDA(m,rand_idx,states=True,labels=labels,savefigure=False)
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
m.state_df.to_pickle(EphysDir + Folder + 'states-corr_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
# #Load previously saved Dataframe from experimental folder
m.state_df = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))



# #check transitions
# Delta = m.Sxx_df.loc[:, 1:4].mean(axis=1)
# Theta = m.Sxx_df.loc[:, 9:10].mean(axis=1)
#
#
# fig, (ax1,ax3) = plt.subplots(2, sharex=True)
# ax1.plot(m.Sxx_df.index,Delta.values,color= 'r', label='raw Delta')
# ax1.grid(False)
# ax1.set_ylabel('Power (Delta raw)')
# ax1.set_xlabel('epochs')
# # ax1.set_title('During SWS')
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# ax2.plot(m.Sxx_df.index,Theta.values,color= 'g', label='Theta')
# ax2.grid(False)
# ax2.set_ylabel('Theta')
# ax3.plot(m.state_df.index,m.state_df['clusters_knn'].values,color= 'b', label='knn')
# ax3.grid(False)
# ax3.set_ylabel('states')
# fig.legend()

####
# Assign states automatically
state_averages = pd.DataFrame()
for state in np.unique(m.state_df['states']):
    print(state)
    state_averages[state] = m.Sxx_df.loc[m.state_df[m.state_df['states']==state].index].mean(axis=0)
    state_averages.to_pickle(state_averages_path)
state_averages['Wake'] = m.Sxx_df.loc[m.state_df[m.state_df['states'].isin(['LTwake','HTwake'])].index].mean(axis=0)
state_averages = pd.read_pickle(state_averages_path)
#create label averages
label_averages = pd.DataFrame()
for label in np.unique(m.state_df['clusters_knn']):
    print(label)
    label_averages[label] = m.Sxx_df.loc[m.state_df[m.state_df['clusters_knn']==label].index].mean(axis=0)

for label in label_averages.iteritems():
    #calculate least square difference for each label
    lsq_df = state_averages.sub(label[1], axis='rows')**2
    print (label[0],lsq_df.mean(axis=0).idxmin())




df['dist'] = np.linalg.norm(df.iloc[:, [1,2]].values - df.iloc[:, [3,4]], axis=1)

# ######################################
# # 5. Use DPC labels to update LDA

#train an LDA based on the ANN labels
lda, X_train = train_lda(Sxx_extended,m.state_df['states'],rand_idx,components=3)
# Create dataframe for LDs
m.LD_df = lda_transform_df(Sxx_extended,lda)


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




