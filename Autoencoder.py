# import tensorflow as tf
# from tf_slim.layers import fully_connected
import pickle
import Config
import seaborn as sns
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras import layers
from plotly.graph_objs import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

import numpy.random as rnd

save_directory = Config.base_path

latent_dim = 20
is_2d_encoded = latent_dim == 2
is_3d_encoded = latent_dim == 3

def load_multiple_mouse_data(file_indexes=None):

    if file_indexes is None:
        file_indexes=[0, 1, 2, 3, 4, 5, 6]

    mouse_files = ['Sxx_norm_200604_m1.pkl',
                   'Sxx_norm_200604_m2.pkl',
                   'Sxx_norm_200604_m3.pkl',
                   'Sxx_norm_200604_m4.pkl',
                   'Sxx_norm_200424_m5.pkl',
                   'Sxx_norm_200424_m6.pkl',
                   'Sxx_norm_200424_m7.pkl']

    directory = Config.base_path + '/Multitaper_df_norm/'

    mouse_data = None

    random_indexes = []

    for file in [mouse_files[i] for i in file_indexes]:
        f = open(directory + file, 'rb')
        m = pickle.load(f)
        rand_idx = np.random.choice(m.shape[0], size=40000, replace=False)
        random_indexes.append(rand_idx)
        m = np.array(m)[rand_idx]
        if mouse_data is None:
            mouse_data = np.empty((0, m.shape[1]))
        mouse_data = np.concatenate((mouse_data, m))

    return mouse_data, random_indexes

def load_multiple_mouse_states(file_indexes, random_indexes):
    mouse_files = ['states_200604_m1.pkl',
                   'states_200604_m2.pkl',
                   'states_200604_m3.pkl',
                   'states_200604_m4.pkl',
                   'states_200424_m5.pkl',
                   'states_200424_m6.pkl',
                   'states_200424_m7.pkl']

    directory = Config.base_path + '/Multitaper_df_norm/'

    state_data = None

    for i, file in enumerate([mouse_files[i] for i in file_indexes]):
        f = open(directory + file, 'rb')
        s = pickle.load(f)
        rand_idx = random_indexes[i]
        s = np.array(s)[rand_idx]
        if state_data is None:
            state_data = np.empty((0, s.shape[1]))
        state_data = np.concatenate((state_data, s))

    return state_data

def create_vae(original_dim, intermediate_dim):

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

    # Create decoder1
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

    return encoder, decoder, vae


def check_labelled_data_on_autoencoder(multitaper_data=None, states=None):

    file_indexes = [0, 1, 2, 3, 4, 5, 6]

    if multitaper_data is None or states is None:
        mouse_data, random_indexes = load_multiple_mouse_data(file_indexes)
        states = load_multiple_mouse_states(file_indexes, random_indexes)
    else:
        mouse_data = multitaper_data
        random_indexes = np.random.choice(mouse_data.shape[0], size=40000, replace=False)
        mouse_data = np.array(mouse_data)[random_indexes]
        states = np.array(states)[random_indexes]

    cluster_indexes = [np.where(states[:, 1] == "HMwake"), np.where(states[:, 1] == "SWS"),
                       np.where(states[:, 1] == "LMwake"), np.where(states[:, 1] == "REM")]

    n = 10

    for index_subset in cluster_indexes:
        plt.figure(figsize=(20, 4))
        rand_idx = np.random.choice(len(index_subset[0]), size=n, replace=False)
        for i, idx in enumerate(index_subset[0][rand_idx]):
            ax = plt.subplot(1, n, i + 1)
            plt.plot(mouse_data[idx])
            plt.gray()
        plt.show()

    original_dim = mouse_data.shape[1]
    intermediate_dim = 512
    batch_size = 128

    encoder, decoder, vae = create_vae(original_dim=original_dim, intermediate_dim=intermediate_dim)

    vae.load_weights(Config.base_path + "/vae_weights_" + str(latent_dim) + "_dimensions.h5")

    #vae.load_weights(Config.base_path + "/vae_weights.h5")

    encoded, _, _ = encoder.predict(mouse_data, batch_size=batch_size)
    decoded = decoder.predict(encoded)

    show_plot = True

    if show_plot:
        if is_2d_encoded:
            plt.figure(figsize=(6, 6))

            sns.scatterplot(encoded[:, 0], encoded[:, 1], hue=states[:, 1],
                                palette=sns.color_palette('hls', 4))
            plt.title("Vassilis' labelled data from LDA")
            plt.show()

            # #embedding = TSNE().fit(encoded)
            # affinities = affinity.PerplexityBasedNN(
            #     encoded,
            #     perplexity=30,
            #     n_jobs=8,
            #     random_state=0,
            # )
            # embedding = TSNEEmbedding(
            #     encoded,
            #     affinities,
            #     learning_rate=1000,
            #     negative_gradient_method="fft",
            #     n_jobs=8,
            #     callbacks=ErrorLogger(),
            #     random_state=42,
            # )
            #
            # embedding1 = embedding.optimize(n_iter=500, exaggeration=12, momentum=0.5)

            plt.figure(figsize=(6, 6))
            #plt.scatter(embedding1[:, 0], embedding1[:, 1])
            plt.scatter(encoded[:, 0], encoded[:, 1])
            plt.colorbar()
            plt.show()

        elif is_3d_encoded:

            fig = plt.figure()
            ax = Axes3D(fig)
            d = {'HMwake': 0,
                      'LMwake': 1,
                      'SWS': 2,
                      'REM': 3,
            }
            n = 40000
            rand_idx = np.random.choice(len(encoded), size=n, replace=False)
            subset = encoded[rand_idx]
            colors = [d[c] for c in np.array(states[rand_idx][:, 1])]
            ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c=colors, alpha=0.1, s=8)
            ax.set_xlabel('ld1')
            ax.set_ylabel('ld2')
            ax.set_zlabel('ld3')
            plt.show()

    cluster_indexes = [np.where(states[:, 1] == "HMwake"), np.where(states[:, 1] == "SWS"),
                       np.where(states[:, 1] == "LMwake"), np.where(states[:, 1] == "REM")]

    n = 10

    for index_subset in cluster_indexes:
        plt.figure(figsize=(20, 4))
        rand_idx = np.random.choice(len(index_subset[0]), size=n, replace=False)
        for i, idx in enumerate(index_subset[0][rand_idx]):
            ax = plt.subplot(1, n, i + 1)
            plt.plot(decoded[idx])
            plt.gray()
        plt.show()

    do_birch_clustering = False

    if do_birch_clustering:
        from numpy import unique
        from numpy import where
        from sklearn.datasets import make_classification
        from sklearn.cluster import Birch

        # define the model
        birch_model = Birch(threshold=0.005, branching_factor=50, n_clusters=4)

        rand_idx = np.random.choice(encoded.shape[0], size=40000, replace=False)
        encoded = encoded[rand_idx]
        decoded = decoded[rand_idx]

        # train the model
        birch_model.fit(encoded)

        # assign each data point to a cluster
        birch_result = birch_model.predict(encoded)

        # get all of the unique clusters
        birch_clusters = unique(birch_result)

        if is_3d_encoded:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.set_xlabel('ld1')
            ax.set_ylabel('ld2')
            ax.set_zlabel('ld3')

        cluster_array = []

        # plot the BIRCH clusters
        for birch_cluster in birch_clusters:
            # get data points that fall in this cluster
            index = where(birch_result == birch_cluster)
            cluster_array.append(index)
            # make the plot
            if is_2d_encoded:
                plt.scatter(encoded[index, 0], encoded[index, 1])
            else:
                ax.scatter(encoded[index, 0], encoded[index, 1], encoded[index, 2])

        # show the BIRCH plot
        plt.show()

        n = 10

        for s in cluster_array:
            plt.figure(figsize=(20, 4))
            for i, idx in enumerate(s[0][0:n]):
                # idx = + randint(0, 12000)
                # Display original
                # Display reconstruction
                ax = plt.subplot(1, n, i + 1)
                plt.plot(decoded[idx])
                plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()

    print("done")

def run_keras_var_autoencoder(load_weights=True, mouse_data=None):
    import keras
    import numpy as np
    from keras import layers
    from sklearn.model_selection import train_test_split

    if mouse_data is None:
        mouse_data, _ = load_multiple_mouse_data()

    original_dim = mouse_data.shape[1]
    intermediate_dim = 512
    batch_size = 128
    epochs = 50

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

    # Create decoder1
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

    x_train, x_test = train_test_split(np.array(mouse_data), test_size=0.33, random_state=42)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    if load_weights:
        vae.load_weights(Config.base_path + "/vae_weights_" + str(latent_dim) + "_dimensions.h5")
    else:
        vae.fit(x_train, x_train,
                epochs=epochs,
                batch_size=256,
                validation_data=(x_test, x_test))
        vae.save_weights(Config.base_path + "/vae_weights_" + str(latent_dim) + "_dimensions.h5")

    x_test_encoded, _, _ = encoder.predict(x_test, batch_size=batch_size)
    decoded = decoder.predict(x_test_encoded)

    if is_2d_encoded:
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
        plt.colorbar()
        plt.show()

    elif is_3d_encoded:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], x_test_encoded[:, 2], c='k', alpha=0.1, s=8)
        ax.set_xlabel('ld1')
        ax.set_ylabel('ld2')
        ax.set_zlabel('ld3')
        plt.show()

    # f = open(Config.base_path + "/encoded_VAE_" + str(latent_dim) + "_dimensions.pkl", "wb")
    #
    #
    #
    # pickle.dump({
    #             "original_data": x_test,
    #             "encoded": x_test_encoded,
    #              "decoded": decoded}, f)

    print("done")


# def run_keras_stacked_autoencoder():
#     # to make this notebook's output stable across runs
#     rnd.seed(42)
#
#     f = open('C:/Source/ClosedLoopEEG/Sxx_norm_mt_41_4.pkl', 'rb')
#     pf = pickle.load(f)
#     x = pf.T[0:40000]
#
#     tf.reset_default_graph()
#
#     n_inputs = x.shape[1]
#     n_hidden1 = 300
#     n_hidden2 = 2  # codings
#     n_hidden3 = n_hidden1
#     n_outputs = n_inputs
#
#     learning_rate = 0.01
#     l2_reg = 0.0001
#
#     initializer = tf.contrib.layers.variance_scaling_initializer()  # He initialization
#     # Equivalent to:
#     # initializer = lambda shape, dtype=tf.float32: tf.truncated_normal(shape, 0., stddev=np.sqrt(2/shape[0]))
#
#     X = tf.placeholder(tf.float32, shape=[None, n_inputs])
#     with tf.contrib.framework.arg_scope(
#             [fully_connected],
#             activation_fn=tf.nn.elu,
#             weights_initializer=initializer,
#             weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)):
#         hidden1 = fully_connected(X, n_hidden1)
#         hidden2 = fully_connected(hidden1, n_hidden2)
#         hidden3 = fully_connected(hidden2, n_hidden3)
#         outputs = fully_connected(hidden3, n_outputs, activation_fn=None)
#
#     mse = tf.reduce_mean(tf.square(outputs - X))
#
#     reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#     loss = tf.add_n([mse] + reg_losses)
#
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     training_op = optimizer.minimize(loss)
#
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
#     n_epochs = 4
#     n_batches = 150
#     batch_size = x.shape[0] / n_batches
#
#     with tf.Session() as sess:
#         init.run()
#         current_start = 0
#         for epoch in range(n_epochs):
#
#             for iteration in range(n_batches):
#                 print("\r{}%".format(100 * iteration // n_batches), end="")
#                 sys.stdout.flush()
#                 X_batch, y_batch = x[current_start:current_start + batch_size]
#                 sess.run(training_op, feed_dict={X: X_batch})
#             mse_train = mse.eval(feed_dict={X: X_batch})
#             print("\r{}".format(epoch), "Train MSE:", mse_train)
#             saver.save(sess, save_directory + "/my_model_all_layers.ckpt")
#
#
def run_keras_basic_autencoder(x, states, load_weights=True):


    import keras
    import numpy as np
    from keras import layers

    states = np.array(states)

    rnd.seed(42)

    # f = open('C:/Source/ClosedLoopEEG/Sxx_norm_mt_41_4.pkl', 'rb')
    # pf = pickle.load(f)
    # x = pf.T[0:40000]


    # This is our input image
    input = keras.Input(shape=(x.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(latent_dim, activation='relu')(input)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(x.shape[1], activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input, decoded)

    encoder = keras.Model(input, encoded)

    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(latent_dim,))
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

    if load_weights:
        autoencoder.load_weights(Config.base_path + "/smoothed_minus_savgol_autoencoder_weights_" + str(latent_dim) + "_dimensions.h5")
    else:
        autoencoder.fit(x_train, x_train,
                        epochs=50,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(x_test, x_test))
        autoencoder.save_weights(Config.base_path + "/smoothed_minus_savgol_autoencoder_weights_" + str(latent_dim) + "_dimensions.h5")

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

    cluster_indexes = [np.where(states[:, 1] == "HMwake"), np.where(states[:, 1] == "SWS"),
                       np.where(states[:, 1] == "LMwake"), np.where(states[:, 1] == "REM")]

    n = 10

    for index_subset in cluster_indexes:
        plt.figure(figsize=(20, 4))
        rand_idx = np.random.choice(len(index_subset[0]), size=n, replace=False)
        for i, idx in enumerate(index_subset[0][rand_idx]):
            ax = plt.subplot(1, n, i + 1)
            plt.plot(decoded[idx])
            plt.gray()
        plt.show()

    d = {'HMwake': 'blue',
         'LMwake': 'green',
         'SWS': 'yellow',
         'REM': 'red',
         }

    if is_3d_encoded:
        fig = plt.figure()
        ax = Axes3D(fig)
        n = 40000
        rand_idx = np.random.choice(len(encoded), size=n, replace=False)
        subset = encoded[rand_idx]
        colors = [d[c] for c in np.array(states)[rand_idx][:, 1]]
        ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c=colors, alpha=0.1, s=8)
        ax.set_xlabel('ld1')
        ax.set_ylabel('ld2')
        ax.set_zlabel('ld3')
        plt.show()
    print("done")
    #plt.plot(x=)


    training_data_states = np.array([s[0] for s in states])

    X_train, X_test, y_train, y_test = train_test_split(decoded, training_data_states, test_size=0.04)

    lda = LDA(n_components=3)
    lda_encoded_data = lda.fit_transform(decoded, training_data_states)
    plot_transformation = True
    if plot_transformation:
        fig = plt.figure()
        ax = Axes3D(fig)
        d = {0: 'blue',#lm
             1: 'yellow',#sws
             2: 'green',#hmw
             3: 'red',#rem
             }
        colors = [d[c] for c in np.array(training_data_states)]
        ax.scatter(lda_encoded_data[:, 0], lda_encoded_data[:, 1], lda_encoded_data[:, 2], c=colors, alpha=0.1, s=8)
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
        ax.set_zlabel('component 3')
        plt.show()

        print("done")
#
# def cluster_on_encoded():
#
#     f = open(Config.base_path + "/encoded_VAE_" + str(latent_dim) + "_dimensions.pkl", 'rb')
#     pf = pickle.load(f)
#     X = pf["encoded"]
#     decoded = pf["decoded"]
#
#     n = 10  # How many digits we will display
#
#     seperate_out_clusters_manually = False
#     manually_labelled_data_file = Config.base_path + "/manually labelled data.pkl"
#
#     display_figures = False
#     do_k_means = False
#     do_birch = True
#     do_gaussian_mix = False
#     do_OPTICS = False
#     do_dbscan = False
#     do_tsne = False
#
#     rand_idx = np.random.choice(X.shape[0], size=40000, replace=False)
#     subset_x = X = X[rand_idx]
#     subset_decoded = decoded = decoded[rand_idx]
#
#     if display_figures:
#         if seperate_out_clusters_manually:
#             rem = [i for i, value in enumerate(X) if value[0] > 0.5 and value[1] < -0.65]
#             sws = [i for i, value in enumerate(X) if value[0] > 0.125 and value[1] > 0]
#             ha = [i for i, value in enumerate(X) if value[0] < -0.25 and value[1] > 0.15]
#             la = [i for i, value in enumerate(X) if value[0] < -0.125 and value[1] < -0.125]
#             f = open(manually_labelled_data_file, 'wb')
#             pickle.dump({
#                 "rem": rem,
#                 "sws": sws,
#                 "ha": ha,
#                 "la": la}, f)
#         else:
#             f = open(manually_labelled_data_file, 'rb')
#             md = pickle.load(f)
#             rem = md["rem"]
#             sws = md["sws"]
#             ha = md["ha"]
#             la = md["la"]
#
#         subsets = [rem, sws, ha, la]
#         for s in subsets:
#             plt.figure(figsize=(20, 4))
#             for i, idx in enumerate(s[0:n]):
#                 # idx = + randint(0, 12000)
#                 # Display original
#                 # Display reconstruction
#                 ax = plt.subplot(1, n, i + 1)
#                 plt.plot(decoded[idx])
#                 plt.gray()
#                 ax.get_xaxis().set_visible(False)
#                 ax.get_yaxis().set_visible(False)
#             plt.show()
#
#     if do_k_means:
#         n_clusters = 5
#
#         from sklearn.cluster import KMeans
#         km4 = KMeans(n_clusters=n_clusters).fit(X)
#
#         if is_2d_encoded:
#             plt.figure(figsize=(12, 8))
#             sns.scatterplot(X[:, 0], X[:, 1], hue=km4.labels_,
#                             palette=sns.color_palette('hls', n_clusters))
#             plt.title('KMeans with 3 Clusters')
#             plt.show()
#         else:
#             fig = plt.figure()
#             ax = Axes3D(fig)
#             ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=km4.labels_, alpha=0.1, s=8)
#             ax.set_xlabel('ld1')
#             ax.set_ylabel('ld2')
#             ax.set_zlabel('ld3')
#             plt.show()
#
#     if do_birch:
#         from numpy import unique
#         from numpy import where
#         from sklearn.datasets import make_classification
#         from sklearn.cluster import Birch
#
#         # define the model
#         birch_model = Birch(threshold=0.005, branching_factor=50, n_clusters=8)
#
#         # train the model
#         birch_model.fit(subset_x)
#
#         # assign each data point to a cluster
#         birch_result = birch_model.predict(subset_x)
#
#         # get all of the unique clusters
#         birch_clusters = unique(birch_result)
#
#         if is_3d_encoded:
#             fig = plt.figure()
#             ax = Axes3D(fig)
#             ax.set_xlabel('ld1')
#             ax.set_ylabel('ld2')
#             ax.set_zlabel('ld3')
#
#         cluster_array = []
#
#         # plot the BIRCH clusters
#         for birch_cluster in birch_clusters:
#             # get data points that fall in this cluster
#             index = where(birch_result == birch_cluster)
#             cluster_array.append(index)
#             # make the plot
#             if is_2d_encoded:
#                 plt.scatter(subset_x[index, 0], subset_x[index, 1])
#             else:
#                 ax.scatter(subset_x[index, 0], subset_x[index, 1], subset_x[index, 2])
#
#         # show the BIRCH plot
#         plt.show()
#
#         for s in cluster_array:
#             plt.figure(figsize=(20, 4))
#             for i, idx in enumerate(s[0][0:n]):
#                 # idx = + randint(0, 12000)
#                 # Display original
#                 # Display reconstruction
#                 ax = plt.subplot(1, n, i + 1)
#                 plt.plot(subset_decoded[idx])
#                 plt.gray()
#                 ax.get_xaxis().set_visible(False)
#                 ax.get_yaxis().set_visible(False)
#             plt.show()
#
#
#     if do_gaussian_mix:
#         from numpy import unique
#         from numpy import where
#         from matplotlib import pyplot
#         from sklearn.datasets import make_classification
#         from sklearn.mixture import GaussianMixture
#
#         # define the model
#         gaussian_model = GaussianMixture(n_components=4, covariance_type='spherical')
#
#         # train the model
#         gaussian_model.fit(X)
#
#         # assign each data point to a cluster
#         gaussian_result = gaussian_model.predict(X)
#
#         # get all of the unique clusters
#         gaussian_clusters = unique(gaussian_result)
#
#         # plot Gaussian Mixture the clusters
#         for gaussian_cluster in gaussian_clusters:
#             # get data points that fall in this cluster
#             index = where(gaussian_result == gaussian_cluster)
#             # make the plot
#             pyplot.scatter(X[index, 0], X[index, 1])
#
#         # show the Gaussian Mixture plot
#         pyplot.show()
#
#     if do_OPTICS:
#         from numpy import unique
#         from numpy import where
#         from matplotlib import pyplot
#         from sklearn.datasets import make_classification
#         from sklearn.cluster import OPTICS
#
#         # define the model
#         optics_model = OPTICS(eps=0.75, min_samples=100)
#
#         # assign each data point to a cluster
#         optics_result = optics_model.fit_predict(X)
#
#         # get all of the unique clusters
#         optics_clusters = unique(optics_result)
#
#         # plot OPTICS the clusters
#         for optics_cluster in optics_clusters:
#             # get data points that fall in this cluster
#             index = where(optics_result == optics_cluster)
#             # make the plot
#             pyplot.scatter(X[index, 0], X[index, 1])
#
#         # show the OPTICS plot
#         pyplot.show()
#
#     if do_dbscan:
#         db = DBSCAN(eps=101, min_samples=10000).fit(X)
#         #
#         plt.figure(figsize=(12, 8))
#         sns.scatterplot(X[:, 0], X[:, 1], db.labels_,
#                     palette=sns.color_palette('hls', np.unique(db.labels_).shape[0]))
#         plt.title('DBSCAN with epsilon 11, min samples 6')
#         plt.show()
#
#     if do_tsne:
#         z_run_tsne = TSNE(perplexity=80, min_grad_norm=1E-12, n_iter=1500).fit_transform(subset_x)
#
#         f = open(Config.base_path + "/VAE_" + str(latent_dim) + "_dimension_tsne.pkl", "wb")
#
#         pickle.dump({
#             "tsne_results": z_run_tsne}, f)
#
#         plt.figure(figsize=(12, 8))
#         sns.scatterplot(z_run_tsne[:, 0], z_run_tsne[:, 1])
#         plt.title('T-SNE')
#         plt.show()
#
#     if False:
        #print("hide comments")
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

        # db scan is pretty useless, too slow
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

    # print("done")

#run_keras_stacked_autoencoder()
#run_keras_basic_autencoder
# run_keras_var_autoencoder(load_weights=False)
# check_labelled_data_on_autoencoder()
#cluster_on_encoded()



