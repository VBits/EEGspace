import ANN
from PlottingUtils import get_random_idx
from Transformations import train_lda
from pydpc import Cluster
from sklearn.neighbors import KNeighborsClassifier

######################################
# 1. Label the multitaper_df using an ANN (use 50 epochs, centered around the epoch of interest)
def chunker_list(seq, size):
    return [seq[i:i + size] for i in range(len(seq) - 1)]
def expand_epochs(dataframe,smoothed_data=False,win=51):
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
    extended_data = pd.DataFrame(data=extended_data, index=dataframe[win//2:-win+win//2].index)
    state_data = m.state_df[win//2:-win+win//2]
    return extended_data, state_data

multitaper_extended, labels_extended = expand_epochs(m.multitaper_df)

rand_idx = get_random_idx(multitaper_extended)

#center the labels
NNetwork = ANN(multitaper_extended,labels_extended,rand_idx)
NNetwork.initialize()
NNetwork.train_model()
NNetwork.get_labels()
NNetwork.save_weights(EphysDir + Folder, 'weights_Multitaper_df_51epochs_centered.h5')
NNetwork.load_weights(EphysDir+Folder,'weights_Sxx_df.h5')
NNetwork.test_accuracy()



######################################
# 2. Use ANN labels for LDA
rand_idx = get_random_idx(m.Sxx_df)

#train an LDA based on the ANN labels
lda, X_train = train_lda(m.Sxx_df,m.state_df['states'],rand_idx,components=3)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_train[:,0],X_train[:,1], X_train[:,2], c='k',alpha=0.1, s=5)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
plt.title('LDA')
plt.savefig(figureFolder+'LDA after original labels' + m.figure_tail, dpi=dpi)

# Create dataframe for LDs
LD = lda.transform(m.Sxx_df)
m.LD_df = pd.DataFrame(data=LD, columns=['LD1', 'LD2','LD3'], index=m.Sxx_df.index)

######################################
# 3 Density peak clustering
# build the density peak clusterer
sample_data = np.ascontiguousarray(m.LD_df.iloc[rand_idx].values)
clu = Cluster(sample_data)
plt.title('DPC', fontsize=15)
plt.savefig(figureFolder+'Density peaks' + m.figure_tail)

# decide the cutoffs for the clusters
clu.assign(400  ,1)
plt.title('DPC boundaries', fontsize=15)
plt.savefig(figureFolder+'Density peak boundaries' + m.figure_tail)

######################################
# 4. Propagate DPC labels
# Use Kneighbors to classify the rest of the data, use odd numbers to avoid draws
clf = KNeighborsClassifier(n_neighbors=201)
clf.fit(sample_data, clu.membership)


### -----
# predict states
m.knn_pred(clf,transform='LDA')

#Assign clusters manually
m.state_df['states'] = 'SWS'
m.state_df.loc[m.state_df['clusters_knn']==3,'states'] = 'LTwake'
m.state_df.loc[m.state_df['clusters_knn']==2,'states'] = 'HTwake'
m.state_df.loc[m.state_df['clusters_knn']==2,'states'] = 'REM'
m.state_df.loc[m.state_df['clusters_knn']==2,'states'] = 'SWS'
m.state_df.loc[m.state_df['clusters_knn']==2,'states'] = 'ambiguous'


### ----
# 3D
# Plot and evaluate state assignment 3D
plot_LDA(m,rand_idx,states=False,alpha=0.5,savefigure=False)
plt.savefig(figureFolder + 'LDA 3D states DPC labels {}_{}'.format(Folder[:6], File[:6]) + m.figure_tail, dpi=dpi)




# Load OR Save knn model
# knn_file = EphysDir+Folder + 'knn_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
knn_file = EphysDir+Folder + 'knn_{}_{}_{}_m{}.joblib'.format(Folder[:6],'210421',m.genotype,m.pos)
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

# #TEMP
# m.state_df.to_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}_{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos,part))
#
# m.state_df = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}_{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos,part))
######################################
# 5. Use DPC labels to update LDA
lda = LDA(n_components=3)
X_train = lda.fit_transform(m.Sxx_norm.iloc[rand_idx], m.state_df['states'][rand_idx])
# X_train = lda.fit_transform(m.Sxx_norm.iloc[rand_idx], state_predictions[rand_idx])
X_train = lda.fit_transform(m.Sxx_norm.iloc[rand_idx], clone.labels_)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_train[:,0],X_train[:,1], X_train[:,2], c='k',alpha=0.1, s=5)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
plt.title('LDA')
plt.savefig(figureFolder+'LDA after DPC labels' + m.figure_tail, dpi=dpi)

# Create dataframe for LDs
LD = lda.transform(m.Sxx_norm)
m.LD_df = pd.DataFrame(data=LD, columns=['LD1', 'LD2','LD3'], index=m.Sxx_norm.index)

### -------------------
# Store or load LDA transformation
lda_filename = EphysDir+Folder + 'lda_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],m.genotype,m.pos)
lda_filename = EphysDir+Folder + 'lda_{}_{}_{}_m{}.joblib'.format(Folder[:6],'210421',m.genotype,m.pos)
lda_filename = EphysDir+Folder + 'lda_210216_210301_Vglut2Cre-SuM_all_mice.joblib'
# Save file
joblib.dump(lda, lda_filename)
# # Recover previously saved file
lda = joblib.load(lda_filename)




