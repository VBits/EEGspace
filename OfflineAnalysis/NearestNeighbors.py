from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def propagate_classes(m,Sxx_extended,rand_idx,state_averages_path,clu,n_neighbors=201):
    # Use Kneighbors to classify the rest of the data, use odd numbers to avoid draws
    sample_data = np.ascontiguousarray(m.LD_df.loc[rand_idx].values)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(sample_data, clu.membership)
    ### -----
    # predict states
    m.knn_pred(clf,Sxx_extended,state_averages_path,transform='LDA')
    return clf
