from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def get_knn_clf(m,rand_idx,clu,n_neighbors=201):
    # Use Kneighbors to classify the rest of the data, use odd numbers to avoid draws
    sample_data = np.ascontiguousarray(m.LD_df.loc[rand_idx].values)
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(sample_data, clu.membership)
    return knn_clf
