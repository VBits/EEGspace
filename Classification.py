import pickle
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from random import *
from time import time
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import metrics

def Classification_Play():
    f = open('C:/Source/ClosedLoopEEG/Sxx_norm_mt_41_4.pkl', 'rb')
    pf = pickle.load(f)

    #plt.hist2d(pf.T, bins=100, normed=False, cmap='plasma')
    #X = pf.values.ravel()
    #plt.hist2d(x=pf[0], y=[*range(0, 201)], bins=100, cmap='plasma')
    #plt.imshow(X=X, Y=range(0, len(X)), cmap='hot', interpolation='nearest')
    #plt.plot(pf[0])
    #plt.show()

    do_tsne = False
    do_isomap = False
    do_umap = False
    do_ann = False
    compare_multitaper = False
    try_autoencoder = True
    load_autoencoder = True

    if compare_multitaper:
        #read in the pkl file
        #take some random epochs and compare them
        #show them in graph
        random_selection = []
        for i in range(0, 10):
            random_num = randint(0, len(pf.T))
            plt.plot(pf[random_num])
        plt.show()

    if do_tsne:
        to_save = {}
        #attempt t-sne
        to_save["pca"] = PCA(n_components=50)
        to_save["pca_results"] = to_save["pca"].fit_transform(pf.T)
        start_time = time.perf_counter()
        to_save["tsne"] = TSNE(n_components=2)
        to_save["tsne_results"] = to_save["tsne"].fit_transform(to_save["pca_results"])
        end_time = time.perf_counter()
        print(str(end_time - start_time) + " seconds for tsne")
        f = open("Classification_data.pkl", 'wb')
        pickle.dump(to_save, f)
        plt.scatter(x=to_save["tsne_results"][:, 0], y=to_save["tsne_results"][:, 1], c='k', linewidths=0, alpha=0.4, s=4)
        plt.show()

    def autoencoder(dims, act='relu', init='glorot_uniform'):
        n_stacks = len(dims) - 1
        # input
        input_img = Input(shape=(dims[0],), name='input')
        x = input_img
        # internal layers in encoder
        for i in range(n_stacks - 1):
            x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

        # hidden layer
        encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(
            x)  # hidden layer, features are extracted from here

        x = encoded
        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

        # output
        x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
        decoded = x
        return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded,
                                                                          name='encoder')

    #try and get a heatmap like what v did
    #plot what a sample of them look like individually

    from scipy.signal import savgol_filter
    def my_filter(x):
        return savgol_filter(x, 41, 1)

    if do_isomap:
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

    if try_autoencoder:
        x = pf.T[0:40000]
        print(x.shape)
        dims = [x.shape[-1], 500, 500, 2000, 4]
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
        pretrain_epochs = 300
        batch_size = 256
        save_dir = 'C:/Source/ClosedLoopEEG/'
        autoencoder, encoder = autoencoder(dims, init=init)
        autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
        autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)  # , callbacks=cb)
        autoencoder.save_weights(save_dir + 'ae_weights.h5')

    if load_autoencoder:
        x = pf.T[0:40000]
        print(x.shape)
        dims = [x.shape[-1], 500, 500, 2000, 4]
        init = VarianceScaling(scale=1. / 3., mode='fan_in',
                               distribution='uniform')
        pretrain_optimizer = SGD(lr=1, momentum=0.9)
        pretrain_epochs = 300
        batch_size = 256
        save_dir = 'C:/Source/ClosedLoopEEG/'
        autoencoder, encoder = autoencoder(dims, init=init)
        autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
        autoencoder.load_weights(save_dir + 'ae_weights.h5')
    #Try UMAP