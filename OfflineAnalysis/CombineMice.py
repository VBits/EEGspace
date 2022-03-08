import os
from natsort import natsorted
import glob
import pandas as pd
import numpy as np

def report_states(EphysDir,Folder):
    os.chdir(EphysDir+Folder)
    print ('testing how many states per mouse')
    for counter, file in enumerate(natsorted(glob.glob("states_*.pkl"))):
        print(file)
        states_file = pd.read_pickle(file)
        print(np.unique(states_file['states'],return_counts=True))

def combine_files(EphysDir,Folder,save=True):
    # Generate a file to be used for models while avoiding overfitting
    os.chdir(EphysDir+Folder)
    if save:
        print ('saving file with states for all the mice')
        Sxx_combined = pd.DataFrame()
        states_combined = pd.DataFrame()
        multitaper_combined = pd.DataFrame()
        for counter, file in enumerate(natsorted(glob.glob("Sxx*.pkl"))):
            print(file)
            Sxx_file = pd.read_pickle(file)
            file_id = file.split("_",2)[-1]
            states_file = pd.read_pickle('states_{}'.format(file_id))
            multitaper_file = pd.read_pickle('Multitaper_df_{}'.format(file_id))
            print(len(Sxx_file),len(multitaper_file),len(states_file))
            # #trim the Sxx file to match the states file length
            # Sxx_file = Sxx_file[:len(states_file)]
            #process with log 10
            # multitaper_file = 10*np.log(multitaper_file[:len(states_file)])
            Sxx_combined = Sxx_combined.append(Sxx_file,ignore_index=True)
            states_combined = states_combined.append(states_file, ignore_index=True)
            multitaper_combined = multitaper_combined.append(multitaper_file, ignore_index=True)
        Sxx_combined.to_pickle(EphysDir + Folder + 'All_mice_Sxx_combined.pkl')
        multitaper_combined.to_pickle(EphysDir + Folder + 'All_mice_multitaper_combined.pkl')
        states_combined.to_pickle(EphysDir + Folder + 'All_mice_states_combined.pkl')

    else:
        print ('loading file with states for all the mice')
        Sxx_combined = pd.read_pickle(EphysDir + Folder + 'All_mice_Sxx_combined.pkl')
        states_combined = pd.read_pickle(EphysDir + Folder + 'All_mice_states_combined.pkl')
        multitaper_combined = pd.read_pickle(EphysDir + Folder + 'All_mice_multitaper_combined.pkl')
    return Sxx_combined, multitaper_combined, states_combined


EphysDir = 'D:/Project_Mouse/Ongoing_analysis/'
Folder = 'Avoid_overfitting/'
#Optional: Check which datasets have 3 states and get rid of them
report_states(EphysDir,Folder)
#combine files to use for the training
Sxx_combined, multitaper_combined,states_combined = combine_files(EphysDir,Folder,save=False)

# get rid of any states that are artifacts
Sxx_combined = Sxx_combined[states_combined['states']!='ambiguous']
multitaper_combined = multitaper_combined[states_combined['states']!='ambiguous']
states_combined = states_combined[states_combined['states']!='ambiguous']

rand_idx = get_random_idx(Sxx_combined,size=200000)
m.state_df = states_combined.loc[rand_idx]
m.Sxx_df = Sxx_combined.loc[rand_idx]
m.multitaper_df = multitaper_combined.loc[rand_idx]


######################################
# 1. Label the multitaper_df using an ANN (use 50 epochs, centered around the epoch of interest)
#Create the Sxx_extended
Sxx_extended = expand_epochs(m,smoothed_data=False)
rand_idx = get_random_idx(Sxx_extended,size=199000)

############################################################
# 2. Get temporary SVM labels
clf = create_svm(m.Sxx_df,m.state_df,rand_idx,return_score=True,save=False)

# Recover previously saved model
svm_clf = load_svm(offline_data_path)
m.svm_labels = get_svm_labels(m.Sxx_df,svm_clf)
m.svm_labels = get_svm_labels(m.multitaper_df,svm_clf)

############################################################
# 3. Train an LDA or load a previously created LDA
# Recover previously saved model
#train an LDA based on the SVM labels
lda, X_train = train_lda(Sxx_extended,m.svm_labels['svm_labels'],rand_idx,components=3)
lda, X_train = train_lda(Sxx_extended,m.state_df['states'],rand_idx,components=3)
# Create dataframe for LDs
m.LD_df = lda_transform_df(Sxx_extended,lda)


#load previously saved LDA
lda_filename = offline_data_path +'lda_extended_211014_211102_Vglut2Cre-CS_m9.joblib'
lda_filename = EphysDir+Folder  + 'lda_extended-corr_210409_210409_B6J_m1.joblib'
lda_filename = EphysDir+Folder + 'lda_extended_210409_210409_B6J_m1.joblib'
lda_filename = EphysDir+Folder + 'lda_combined_iter=1_win21_polyn4_dpc_labels.joblib'
lda = joblib.load(lda_filename)
# Create dataframe for LDs
m.LD_df = lda_transform_df(Sxx_extended,lda)
#Evaluate SVM labels
plot_LDA(m,rand_idx,m.svm_labels['svm_labels'],savefigure=False)
plot_LDA(m,rand_idx,m.state_df['states'],savefigure=False)
#Or evaluate LDA shape seperately
title= 'LDA no labels', title= 'LDA svm labels'
plot_LDA(m,rand_idx,savefigure=False)
plt.savefig(m.figureFolder+title + m.figure_tail, dpi=dpi)

######################################
# 4. Density peak clustering
# build the density peak clusterer
clu = get_dpc(m,rand_idx,savefigure=True)
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
plot_LDA(m,rand_idx,m.state_df['states'],savefigure=False)
plt.savefig(m.figureFolder + 'LDA DPC labels {}_{}'.format(Folder[:6], File[:6]) + m.figure_tail, dpi=dpi)

