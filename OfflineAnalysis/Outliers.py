"""
This code can be used to annotate outliers using DBSCAN
These labels can then be propagated using KNN to the rest of the dataset
"""
#################
#TODO test and color outlier
from sklearn.cluster import DBSCAN

dbscan_model = DBSCAN(eps=2.9,min_samples=100).fit(m.LD_df.loc[rand_idx]) # (2, 100)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(m.LD_df['LD1'].loc[rand_idx],m.LD_df['LD2'].loc[rand_idx],m.LD_df['LD3'].loc[rand_idx],c=dbscan_model.labels_,cmap='Dark2_r',alpha=0.5, s=5)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
plt.title('LDA')
plt.savefig(figureFolder+'LDA and labeled outliers' + m.figure_tail, dpi=dpi)

np.unique(dbscan_model.labels_,return_counts=True)


clf_outlier = KNeighborsClassifier(n_neighbors=5)
sample_data = np.ascontiguousarray(m.LD_df.loc[rand_idx].values)
clf_outlier.fit(sample_data, dbscan_model.labels_)


### -----
# predict states
m.state_df['outliers'] = clf_outlier.predict(m.LD_df)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(m.LD_df['LD1'].loc[rand_idx],m.LD_df['LD2'].loc[rand_idx],m.LD_df['LD3'].loc[rand_idx],c=m.state_df['outliers'].loc[rand_idx],cmap='bwr',alpha=0.5, s=5)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
plt.title('LDA')
plt.savefig(figureFolder+'LDA and KNN outliers' + m.figure_tail, dpi=dpi)

np.unique(m.state_df['outliers'],return_counts=True)




# Annotate the state dataframe
m.state_df.loc[m.state_df['outliers']!=0,'states']= 'ambiguous'

np.unique(m.state_df['states'],return_counts=True)

#TEMP
m.state_df.to_pickle(EphysDir + Folder + 'states_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
