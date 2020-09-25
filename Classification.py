import pickle
import numpy as np
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

def Classification_Play():
    f = open('C:/Source/ClosedLoopEEG/Sxx_norm.pkl', 'rb')
    pf = pickle.load(f)

    #plt.hist2d(pf.T, bins=100, normed=False, cmap='plasma')
    #X = pf.values.ravel()
    #plt.hist2d(x=pf[0], y=[*range(0, 201)], bins=100, cmap='plasma')
    #plt.imshow(X=X, Y=range(0, len(X)), cmap='hot', interpolation='nearest')
    #plt.plot(pf[0])
    #plt.show()

    # to_save = {}
    # #attempt t-sne
    # to_save["pca"] = PCA(n_components=50)
    # to_save["pca_results"] = to_save["pca"].fit_transform(pf.T)
    # start_time = time.perf_counter()
    # to_save["tsne"] = TSNE(n_components=2)
    # to_save["tsne_results"] = to_save["tsne"].fit_transform(to_save["pca_results"])
    # end_time = time.perf_counter()
    # print(str(end_time - start_time) + " seconds for tsne")
    # f = open("Classification_data.pkl", 'wb')
    # pickle.dump(to_save, f)
    # plt.scatter(x=to_save["tsne_results"][:, 0], y=to_save["tsne_results"][:, 1], c='k', linewidths=0, alpha=0.4, s=4)
    # plt.show()

    #try and get a heatmap like what v did
    #plot what a sample of them look like individually

    from scipy.signal import savgol_filter
    def my_filter(x):
        return savgol_filter(x, 41, 1)

    #Attempt to do isomap
    to_save = {}
    # attempt t-sne

    X = pf.T
    for i in range(0,10):
        X = X.apply(my_filter)
    # plt.plot(X.T[0])
    # plt.show()
    to_save["pca"] = PCA(n_components=5)
    to_save["pca_results"] = to_save["pca"].fit_transform(X[0:20000])
    start_time = time.perf_counter()
    to_save["tsne"] = Isomap(n_components=2)
    to_save["tsne_results"] = to_save["tsne"].fit_transform(to_save["pca_results"])
    end_time = time.perf_counter()
    print(str(end_time - start_time) + " seconds for tsne")
    f = open("Classification_data.pkl", 'wb')
    pickle.dump(to_save, f)
    plt.scatter(x=to_save["tsne_results"][:, 0], y=to_save["tsne_results"][:, 1], c='k', linewidths=0, alpha=0.4, s=4)
    plt.show()

    #Try UMAP