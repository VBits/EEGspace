import tensorflow as tf
from tf_slim.layers import fully_connected
import pickle
import Config
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import numpy as np

import numpy.random as rnd

save_directory = Config.base_path

def run_keras_stacked_autoencoder():
    # to make this notebook's output stable across runs
    rnd.seed(42)

    f = open('C:/Source/ClosedLoopEEG/Sxx_norm_mt_41_4.pkl', 'rb')
    pf = pickle.load(f)
    x = pf.T[0:40000]

    tf.reset_default_graph()

    n_inputs = x.shape[1]
    n_hidden1 = 300
    n_hidden2 = 2  # codings
    n_hidden3 = n_hidden1
    n_outputs = n_inputs

    learning_rate = 0.01
    l2_reg = 0.0001

    initializer = tf.contrib.layers.variance_scaling_initializer()  # He initialization
    # Equivalent to:
    # initializer = lambda shape, dtype=tf.float32: tf.truncated_normal(shape, 0., stddev=np.sqrt(2/shape[0]))

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            activation_fn=tf.nn.elu,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)):
        hidden1 = fully_connected(X, n_hidden1)
        hidden2 = fully_connected(hidden1, n_hidden2)
        hidden3 = fully_connected(hidden2, n_hidden3)
        outputs = fully_connected(hidden3, n_outputs, activation_fn=None)

    mse = tf.reduce_mean(tf.square(outputs - X))

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([mse] + reg_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    n_epochs = 4
    n_batches = 150
    batch_size = x.shape[0] / n_batches

    with tf.Session() as sess:
        init.run()
        current_start = 0
        for epoch in range(n_epochs):

            for iteration in range(n_batches):
                print("\r{}%".format(100 * iteration // n_batches), end="")
                sys.stdout.flush()
                X_batch, y_batch = x[current_start:current_start + batch_size]
                sess.run(training_op, feed_dict={X: X_batch})
            mse_train = mse.eval(feed_dict={X: X_batch})
            print("\r{}".format(epoch), "Train MSE:", mse_train)
            saver.save(sess, save_directory + "/my_model_all_layers.ckpt")


def run_keras_basic_autencoder():

    import keras
    import numpy as np
    from keras import layers

    rnd.seed(42)

    f = open('C:/Source/ClosedLoopEEG/Sxx_norm_mt_41_4.pkl', 'rb')
    pf = pickle.load(f)
    x = pf.T[0:40000]

    # This is the size of our encoded representations
    encoding_dim = 2

    # This is our input image
    input = keras.Input(shape=(x.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(x.shape[1], activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input, decoded)

    encoder = keras.Model(input, encoded)

    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    x_train = x_test = np.array(x)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    autoencoder.fit(x_train, x_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded = encoder.predict(x_test)
    decoded = decoder.predict(encoded)

    n = 10  # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        idx = i + 200
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.plot(x_test[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.plot(decoded[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    #plt.plot(x=)


def run_keras_var_autoencoder():
    import keras
    import numpy as np
    from keras import layers
    from sklearn.model_selection import train_test_split

    f = open('C:/Source/ClosedLoopEEG/Sxx_norm_mt_41_4.pkl', 'rb')
    pf = pickle.load(f)
    pf = pf.T[0:40000]

    original_dim = pf.shape[1]
    intermediate_dim = 512
    batch_size = 128
    latent_dim = 2
    epochs = 100

    inputs = keras.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_sigma = layers.Dense(latent_dim)(h)

    from keras import backend as K

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_sigma])

    # Create encoder
    encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

    # Create decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    x_train, x_test = train_test_split(np.array(pf), test_size=0.33, random_state=42)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    vae.fit(x_train, x_train,
            epochs=epochs,
            batch_size=256,
            validation_data=(x_test, x_test))

    x_test_encoded, _, _ = encoder.predict(x_test, batch_size=batch_size)
    decoded = decoder.predict(x_test_encoded)

    if True:
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1]) #, c=y_test)
        plt.colorbar()
        plt.show()

    vae.save_weights('vae_weights.h5')

    f = open("encoded_VAE.pkl", 'wb')
    pickle.dump({
                "original_data": x_test,
                "encoded": x_test_encoded,
                 "decoded": decoded}, f)

    print("done")


def cluster_on_encoded():

    f = open("encoded_VAE.pkl", 'rb')
    pf = pickle.load(f)
    X = pf["encoded"]
    decoded = pf["decoded"]

    n = 10  # How many digits we will display
    #plt.figure(figsize=(20, 4))
    from random import randint

    rem = [i for i, value in enumerate(X) if value[0] > 0.5 and value[1] < -0.65]
    sws = [i for i, value in enumerate(X) if value[0] > 0.125 and value[1] > 0]
    ha = [i for i, value in enumerate(X) if value[0] < -0.25 and value[1] > 0.15]
    la = [i for i, value in enumerate(X) if value[0] < -0.125 and value[1] < -0.125]

    f = open("manually labelled data.pkl", 'wb')
    pickle.dump({
        "rem": rem,
        "sws": sws,
        "ha": ha,
        "la": la}, f)

    subsets = [rem, sws, ha, la]
    for s in []:#subsets:
        for i, idx in enumerate(s[0:n]):
            # idx = + randint(0, 12000)
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.plot(X[idx])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            # encoded = X[idx]
            # encoded_array = []
            # for e in encoded:
            #     encoded_array.append(e)
            # decoded = decoder.predict(np.array([encoded_array]))
            plt.plot(decoded[idx])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    #db scan is pretty useless
    # db = DBSCAN(eps=101, min_samples=10000).fit(X)
    #
    # plt.figure(figsize=(12, 8))
    # sns.scatterplot(X[:, 0], X[:, 1], db.labels_,
    #             palette=sns.color_palette('hls', np.unique(db.labels_).shape[0]))
    # plt.title('DBSCAN with epsilon 11, min samples 6')
    # plt.show()

    # from sklearn.cluster import AgglomerativeClustering
    # n_clusters = 4
    #
    # agglom = AgglomerativeClustering(n_clusters=n_clusters, linkage='average').fit(X)
    #
    # plt.figure(figsize=(12, 8))
    # sns.scatterplot(X[:, 0], X[:, 1], hue=agglom.labels_,
    #                 palette=sns.color_palette('hls', n_clusters))
    # plt.title('Agglomerative with 4 Clusters')
    # plt.show()

    # from sklearn.cluster import MeanShift, estimate_bandwidth
    # # The following bandwidth can be automatically detected using
    # #bandwidth = estimate_bandwidth(X, quantile=0.1)
    # ms = MeanShift(min_bin_freq=1000).fit(X)
    #
    # plt.figure(figsize=(12, 8))
    # sns.scatterplot(X[:, 0], X[:, 1], ms.labels_,
    #                 palette=sns.color_palette('hls', np.unique(ms.labels_).shape[0]))
    # plt.plot()
    # plt.title('MeanShift')
    # plt.show()

    if True:
        n_clusters = 10

        from sklearn.cluster import KMeans
        km4 = KMeans(n_clusters=n_clusters).fit(X)

        plt.figure(figsize=(12, 8))
        sns.scatterplot(X[:, 0], X[:, 1], hue=km4.labels_,
                        palette=sns.color_palette('hls', n_clusters))
        plt.title('KMeans with 3 Clusters')
        plt.show()

    # from numpy import unique
    # from numpy import where
    # from matplotlib import pyplot
    # from sklearn.datasets import make_classification
    # from sklearn.cluster import AffinityPropagation
    #
    # # define the model
    # model = AffinityPropagation(verbose=True)
    #
    # # train the model
    # model.fit(X)
    #
    # # assign each data point to a cluster
    # result = model.predict(X)
    #
    # # get all of the unique clusters
    # clusters = unique(result)
    #
    # # plot the clusters
    # for cluster in clusters:
    #     # get data points that fall in this cluster
    #     index = where(result == cluster)
    #     # make the plot
    #     pyplot.scatter(X[index, 0], X[index, 1])
    #
    # # show the plot
    # pyplot.show()

    if True:
        from numpy import unique
        from numpy import where
        from matplotlib import pyplot
        from sklearn.datasets import make_classification
        from sklearn.cluster import Birch

        # define the model
        birch_model = Birch(threshold=0.005, branching_factor=50, n_clusters=4)

        # train the model
        birch_model.fit(X)

        # assign each data point to a cluster
        birch_result = birch_model.predict(X)

        # get all of the unique clusters
        birch_clusters = unique(birch_result)

        # plot the BIRCH clusters
        for birch_cluster in birch_clusters:
            # get data points that fall in this cluster
            index = where(birch_result == birch_cluster)
            # make the plot
            pyplot.scatter(X[index, 0], X[index, 1])

        # show the BIRCH plot
        pyplot.show()

    if False:
        from numpy import unique
        from numpy import where
        from matplotlib import pyplot
        from sklearn.datasets import make_classification
        from sklearn.mixture import GaussianMixture

        # define the model
        gaussian_model = GaussianMixture(n_components=4, covariance_type='spherical')

        # train the model
        gaussian_model.fit(X)

        # assign each data point to a cluster
        gaussian_result = gaussian_model.predict(X)

        # get all of the unique clusters
        gaussian_clusters = unique(gaussian_result)

        # plot Gaussian Mixture the clusters
        for gaussian_cluster in gaussian_clusters:
            # get data points that fall in this cluster
            index = where(gaussian_result == gaussian_cluster)
            # make the plot
            pyplot.scatter(X[index, 0], X[index, 1])

        # show the Gaussian Mixture plot
        pyplot.show()

    if False:
        from numpy import unique
        from numpy import where
        from matplotlib import pyplot
        from sklearn.datasets import make_classification
        from sklearn.cluster import OPTICS

        # define the model
        optics_model = OPTICS(eps=0.75, min_samples=100)

        # assign each data point to a cluster
        optics_result = optics_model.fit_predict(X)

        # get all of the unique clusters
        optics_clusters = unique(optics_result)

        # plot OPTICS the clusters
        for optics_cluster in optics_clusters:
            # get data points that fall in this cluster
            index = where(optics_result == optics_cluster)
            # make the plot
            pyplot.scatter(X[index, 0], X[index, 1])

        # show the OPTICS plot
        pyplot.show()

    print("done")


#run_keras_stacked_autoencoder()
#run_keras_basic_autencoder
run_keras_var_autoencoder()
cluster_on_encoded()



