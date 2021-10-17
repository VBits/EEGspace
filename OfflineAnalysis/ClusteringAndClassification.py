from ANN import get_ANN_labels
from PlottingUtils import get_random_idx
from Transformations import train_lda

######################################
# 1. Get temp labels from ANN
ann_labels = get_ANN_labels(mh.Sxx_df)
######################################
# 2. Use ANN labels for LDA
rand_idx = get_random_idx(mh.Sxx_df)

#train an LDA based on the ANN labels
X_train = train_lda(mh.Sxx_df,ann_labels,components=3)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_train[:,0],X_train[:,1], X_train[:,2], c='k',alpha=0.1, s=5)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
plt.title('LDA')
plt.savefig(figureFolder+'LDA after ANN labels' + mh.figure_tail, dpi=dpi)

# Create dataframe for LDs
LD = lda.transform(mh.Sxx_norm)
mh.LD_df = pd.DataFrame(data=LD, columns=['LD1', 'LD2','LD3'], index=mh.Sxx_norm.index)

######################################
# 3 Density peak clustering
# build the density peak clusterer
sample_data = np.ascontiguousarray(mh.LD_df.iloc[rand_idx].values)
# sample_data = np.ascontiguousarray(mh.LD_df.iloc[rand_idx][mh.state_df.iloc[rand_idx]['outliers']==0].values)
# sample_data = np.ascontiguousarray(mh.pC_df.iloc[rand_idx].values)
sample_data = np.ascontiguousarray(LD_df_test.values)
clu = Cluster(sample_data)
plt.title('DPC', fontsize=15)
plt.savefig(figureFolder+'Density peaks' + mh.figure_tail)

# decide the cutoffs for the clusters
clu.assign(200,2)
plt.title('DPC boundaries', fontsize=15)
plt.savefig(figureFolder+'Density peak boundaries' + mh.figure_tail)

######################################
# 4. Propagate DPC labels
# Use Kneighbors to classify the rest of the data, use odd numbers to avoid draws
clf = KNeighborsClassifier(n_neighbors=201)
clf.fit(sample_data, clu.membership)
clf.fit(sample_data,clone.labels_)
clf.fit(sample_data,state_predictions[rand_idx])
clf.fit(sample_data, mh.state_df['clusters_knn'].iloc[rand_idx].values)

### -----
# predict states
mh.knn_pred(clf,transform='LDA')
# mh.knn_pred(clf,transform='PCA')

#Assign clusters manually
mh.state_df['states'] = 'SWS'
mh.state_df.loc[mh.state_df['clusters_knn']==3,'states'] = 'LTwake'
mh.state_df.loc[mh.state_df['clusters_knn']==2,'states'] = 'HTwake'
mh.state_df.loc[mh.state_df['clusters_knn']==2,'states'] = 'REM'
mh.state_df.loc[mh.state_df['clusters_knn']==2,'states'] = 'SWS'
mh.state_df.loc[mh.state_df['clusters_knn']==2,'states'] = 'ambiguous'


### ----
# 3D
# Plot and evaluate state assignment 3D
fig = plt.figure()
ax = Axes3D(fig)
if 'states' in mh.state_df:
    ax.scatter(mh.LD_df['LD1'][rand_idx],mh.LD_df['LD2'][rand_idx], mh.LD_df['LD3'][rand_idx],
               c=mh.state_df['states'][rand_idx].apply(lambda x: mh.colors[x]),alpha=0.5, s=5,linewidths=0)
else:
    ax.scatter(mh.LD_df['LD1'][rand_idx], mh.LD_df['LD2'][rand_idx], mh.LD_df['LD3'][rand_idx],
               c=mh.state_df['clusters_knn'][rand_idx],cmap='tab10', alpha=0.5, s=5, linewidths=0)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
plt.title('KNN')
plt.savefig(figureFolder+'LDA 3D states DPC labels {}_{}'.format(Folder[:6],File[:6]) + mh.figure_tail, dpi=dpi)

#TODO OPTIONAL
fig = plt.figure()
ax = Axes3D(fig)
# ax.scatter(mh.pC_df['pC1'][rand_idx],mh.pC_df['pC2'][rand_idx], mh.pC_df['pC3'][rand_idx],
#                c=clu.membership,cmap='Accent',alpha=0.5, s=5,linewidths=0)
# ax.scatter(mh.LD_df['LD1'][rand_idx],mh.LD_df['LD2'][rand_idx], mh.LD_df['LD3'][rand_idx],
#                c=clu.membership,cmap='Accent',alpha=0.5, s=5,linewidths=0)
ax.scatter(mh.LD_df['LD1'][rand_idx],mh.LD_df['LD2'][rand_idx], mh.LD_df['LD3'][rand_idx],
               c=state_predictions[rand_idx],cmap='Accent',alpha=0.5, s=5,linewidths=0)
# ax.scatter(mh.LD_df['LD1'][rand_idx],mh.LD_df['LD2'][rand_idx], mh.LD_df['LD3'][rand_idx],
#                c=mh.state_df['clusters_knn'][rand_idx],cmap='Accent',alpha=0.5, s=5,linewidths=0)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
plt.savefig(figureFolder+'PCA 3D states' + mh.figure_tail, dpi=dpi)

np.unique(mh.state_df['states'],return_counts=True)

mh.state_df['states'][mh.state_df['states']=='HTwake'] = 'REMi'
# mh.state_df['states'][mh.state_df['states']=='SWS'] = 'HTwake'
mh.state_df['states'][mh.state_df['states']=='Wake'] = 'LTwake'
mh.state_df['states'][mh.state_df['states']=='REM'] = 'HTwake'
# mh.state_df['states'][mh.state_df['states']=='REMi'] = 'REM'
mh.state_df['states'][mh.state_df['states']=='REMi'] = 'REM'



# Load OR Save knn model
# knn_file = EphysDir+Folder + 'knn_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],mh.genotype,mh.pos)
knn_file = EphysDir+Folder + 'knn_{}_{}_{}_m{}.joblib'.format(Folder[:6],'210421',mh.genotype,mh.pos)
# knn_file = EphysDir+Folder + 'knn_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],mh.genotype,'all_mice')
### -------------------
# Save file
joblib.dump(clf, knn_file)
### -------------------
# Recover previously saved file
clf = joblib.load(knn_file)

### -------------------
#Save State Dataframe
mh.state_df.to_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],mh.genotype,mh.pos))
# #Load previously saved Dataframe from experimental folder
mh.state_df = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],mh.genotype,mh.pos))

# #TEMP
# mh.state_df.to_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}_{}.pkl'.format(Folder[:6],File[:6],mh.genotype,mh.pos,part))
#
# mh.state_df = pd.read_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}_{}.pkl'.format(Folder[:6],File[:6],mh.genotype,mh.pos,part))
######################################
# 5. Use DPC labels to update LDA
lda = LDA(n_components=3)
X_train = lda.fit_transform(mh.Sxx_norm.iloc[rand_idx], mh.state_df['states'][rand_idx])
# X_train = lda.fit_transform(mh.Sxx_norm.iloc[rand_idx], state_predictions[rand_idx])
X_train = lda.fit_transform(mh.Sxx_norm.iloc[rand_idx], clone.labels_)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_train[:,0],X_train[:,1], X_train[:,2], c='k',alpha=0.1, s=5)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
plt.title('LDA')
plt.savefig(figureFolder+'LDA after DPC labels' + mh.figure_tail, dpi=dpi)

# Create dataframe for LDs
LD = lda.transform(mh.Sxx_norm)
mh.LD_df = pd.DataFrame(data=LD, columns=['LD1', 'LD2','LD3'], index=mh.Sxx_norm.index)

### -------------------
# Store or load LDA transformation
lda_filename = EphysDir+Folder + 'lda_{}_{}_{}_m{}.joblib'.format(Folder[:6],File[:6],mh.genotype,mh.pos)
lda_filename = EphysDir+Folder + 'lda_{}_{}_{}_m{}.joblib'.format(Folder[:6],'210421',mh.genotype,mh.pos)
lda_filename = EphysDir+Folder + 'lda_210216_210301_Vglut2Cre-SuM_all_mice.joblib'
# Save file
joblib.dump(lda, lda_filename)
# # Recover previously saved file
lda = joblib.load(lda_filename)

######################################
# 6. Optional step: Repeat DPC if not happy



######################################
# 7. Plot density per state
if 'states' in mh.state_df:
    labels = 'states'
else:
    labels = 'clusters_knn'
# for state in ['HMwake', 'LMwake', 'REM', 'SWS']:
for state in np.unique(mh.state_df[labels]):
    print(state)
    b = np.array([mh.state_df[labels].values == state])
    #select two wake states
        # b = mh.state_df['4_states'].isin(['HMwake', 'LMwake']).values
    Sxx_state = mh.Sxx_norm.iloc[b.flatten()]
    density_state, bins_state = density_calc(Sxx_state.T, boundary=(-25, 80))

    plt.figure()
    normalize = mpl.colors.Normalize(vmin=0, vmax=0.185)
    # normalize = mpl.colors.LogNorm(vmin=0.001,vmax=0.04)
    p = plt.pcolormesh(mh.Sxx_norm.columns, bins_state, density_state.T, cmap='plasma', norm=normalize)
    plt.title('Power Density Distribution - {}'.format(state))
    plt.ylabel('Power [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.show()
    cbar = plt.colorbar(p, label='Density')
    # plt.xlim(20, 50)
    # plt.xlim(0, 15)
    plt.ylim(-10, 80)
    plt.savefig(figureFolder + 'Power Density Distribution - {} LDA states 4sec 1iter 9window smoothing on the req axis no baseline substraction'.format(state) + mh.figure_tail, dpi=dpi)

