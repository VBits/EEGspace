from sklearn.neighbors import KNeighborsClassifier

def propagate_classes(m,rand_idx,clu,n_neighbors=201):
    # Use Kneighbors to classify the rest of the data, use odd numbers to avoid draws
    sample_data = np.ascontiguousarray(m.LD_df.iloc[rand_idx].values)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(sample_data, clu.membership)
    ### -----
    # predict states
    m.knn_pred(clf,transform='LDA')
    return clf
