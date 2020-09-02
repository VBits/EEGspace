###################################################
# Density peak clustering -------------------------------------------------------------
# build the density peak clusterer
clu = Cluster(mh.pC[rand_idx])
plt.title('DPC', fontsize=15)
plt.savefig(figureFolder+'Density peaks' + figure_tail)

# decide the cutoffs for the clusters
clu.assign(100, 0.1)
plt.title('DPC boundaries', fontsize=15)
plt.savefig(figureFolder+'Density peak boundaries' + figure_tail)

def plots_DPC():
    matplotlib.use('Agg')
    # plot cluster centers, density map and cluster membership to assess the results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].set_title('Density plot')
    ax[0].scatter(*mh.pC.T[:,rand_idx], s=4, c=clu.density, alpha=0.4, cmap='viridis')
    ax[1].set_title('Density peaks')
    ax[1].scatter(*mh.pC.T[:,rand_idx], c='k', linewidths=0, s=4, alpha=0.4)
    ax[1].scatter(mh.pC[rand_idx,0][clu.clusters], mh.pC[rand_idx,1][clu.clusters], s=10, c='red')
    ax[2].set_title('Cluster membership')
    ax[2].scatter(*mh.pC.T[:,rand_idx], s=4, c=clu.membership, alpha=0.4, cmap='Accent')
    for _ax in ax:
        _ax.set_xlabel("PC1", fontsize=12)
        _ax.set_ylabel("PC2", fontsize=12)
        _ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(figureFolder+'Density peak clustering' + figure_tail, dpi=dpi)


    # plot boundaries between clusters
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].set_title('Border members')
    ax[0].scatter(*mh.pC.T[:,rand_idx], s=4, c=clu.membership, alpha=0.4, cmap='Accent')
    ax[0].scatter(mh.pC[rand_idx,0][clu.border_member],mh.pC[rand_idx,1][clu.border_member], s=4, c="red", alpha = 0.2)
    ax[1].set_title('Core members')
    ax[1].scatter(mh.pC[rand_idx,0][clu.core_idx], mh.pC[rand_idx,1][clu.core_idx],s=4, c=clu.membership[clu.core_idx], alpha=0.4, cmap='Accent')
    ax[1].scatter(mh.pC[rand_idx,0][clu.halo_idx],mh.pC[rand_idx,1][clu.halo_idx], s=4, c="red", alpha = 0.2)
    for _ax in ax:
        _ax.set_xlabel("PC1", fontsize=12)
        _ax.set_ylabel("PC2", fontsize=12)
        _ax.set_aspect('equal')
    fig.tight_layout()
    plt.savefig(figureFolder+'Density peak boundaries and cores' + figure_tail, dpi=dpi)
    matplotlib.use('qt5Agg')
plots_DPC()

### -------------------
# Use Kneighbors to classify the rest of the data, use odd numbers to avoid
clf = KNeighborsClassifier(n_neighbors=201)
clf.fit(mh.pC[rand_idx], clu.membership)


# Load OR Save svm model
model_filename = EphysDir + Folder + '{} knn {}.joblib'.format(mh,date)

### -------------------
# Save file
joblib.dump(clf, model_filename)
### -------------------
# Recover previously saved file
clf = joblib.load(model_filename)

### -------------------
# predict states
mh.knn_pred(clf)

### -------------------
# Plot and evaluate state assignment
fig = plt.figure()
plt.scatter(*mh.pC.T[:,rand_idx], c=mh.state_df['3_states'][rand_idx].apply(lambda x: mh.colors[x]), linewidths=0,
            s=5, alpha=0.5)
plt.title('KNN')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig(figureFolder+'KNN' + figure_tail, dpi=dpi)