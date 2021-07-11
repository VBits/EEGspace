from smrx_version import Offline_analysis
from Autoencoder import *
import Storage
import keras
import numpy as np
from keras import layers
from sklearn.model_selection import train_test_split
from keras.regularizers import l1


def try_output_smoothed(multitaper, smoothed, states, load_weights=False):

    states = np.array(states)
    multitaper = np.array(multitaper)

    rnd.seed(42)

    latent_dim = 3

    # This is our input image
    input = keras.Input(shape=(multitaper.shape[1],))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(latent_dim, activation='relu')(input)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(smoothed.shape[1], activation='linear')(encoded) #, activity_regularizer=l1(0.001)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input, decoded)

    encoder = keras.Model(input, encoded)

    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(latent_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    X = np.array(multitaper)
    y = np.array(smoothed)

    X = X.astype('float32')
    y = y.astype('float32')
    # X = X.reshape((len(X), np.prod(X.shape[1:])))
    # y = y.reshape((len(y), np.prod(y.shape[1:])))
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    if load_weights:
        autoencoder.load_weights(
            Config.base_path + "/multitaper_to_smoothed_weights_" + str(latent_dim) + "_dimensions.h5")
    else:
        autoencoder.fit(X_train, y_train,
                        epochs=50,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(X_test, y_test))
        autoencoder.save_weights(
            Config.base_path + "/multitaper_to_smoothed_weights_" + str(latent_dim) + "_dimensions.h5")

    # Encode and decode some digits
    # Note that we take them from the *test* set
    encoded = encoder.predict(X_test)
    decoded = decoder.predict(encoded)

    #_, accuracy = autoencoder.evaluate(X_train, y_train)
    #print(accuracy * 100)

    if True:
        n = 10  # How many digits we will display
        plt.figure(figsize=(30, 6))
        for i in range(n):
            idx = i + 200

            # Display original
            ax = plt.subplot(3, n, i + 1)
            plt.plot(X_test[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display original
            ax = plt.subplot(3, n, i + 1 + n)
            plt.plot(y_test[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(3, n, i + 1 + (2 * n))
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
            result = autoencoder.predict(X[idx].reshape(1, X.shape[1]))
            plt.plot(result[0])
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
    # plt.plot(x=)

    training_data_states = np.array([s[0] for s in states])

    decoded = autoencoder.predict(X)

    lda = LDA(n_components=3)
    lda_encoded_data = lda.fit_transform(decoded, training_data_states)
    plot_transformation = True
    if plot_transformation:
        fig = plt.figure()
        ax = Axes3D(fig)
        d = {0: 'blue',  # lm
             1: 'yellow',  # sws
             2: 'green',  # hmw
             3: 'red',  # rem
             }
        colors = [d[c] for c in np.array(training_data_states)]
        ax.scatter(lda_encoded_data[:, 0], lda_encoded_data[:, 1], lda_encoded_data[:, 2], c=colors, alpha=0.1, s=8)
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
        ax.set_zlabel('component 3')
        plt.show()

    print("done")


def try_predict_class(multitaper, states, load_weights=False):

    states = np.array(states)
    training_data_states = np.array([s[0] for s in states])
    multitaper = np.array(multitaper)

    rnd.seed(42)

    latent_dim = 3

    num_classes = len(np.unique(training_data_states))

    X = np.array(multitaper)
    y = np.array(training_data_states)

    X = X.astype('float32')
    y = y.astype('float32')
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # X = X.reshape((len(X), np.prod(X.shape[1:])))
    # y = y.reshape((len(y), np.prod(y.shape[1:])))
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    def make_model(input_shape):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    model = make_model(input_shape=X_train.shape[1:])
    keras.utils.plot_model(model, show_shapes=True)

    epochs = 500
    batch_size = 32

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    model = keras.models.load_model("best_model.h5")

    test_loss, test_acc = model.evaluate(X_test, y_test)

    print("Test accuracy", test_acc)
    print("Test loss", test_loss)

    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()

    # if load_weights:
    #     model.load_weights(
    #         Config.base_path + "/multitaper_to_smoothed_weights_" + str(latent_dim) + "_dimensions.h5")
    # else:
    #     model.fit(X_train, y_train,
    #                     epochs=50,
    #                     batch_size=256,
    #                     shuffle=True,
    #                     validation_data=(X_test, y_test))
    #     model.save_weights(
    #         Config.base_path + "/multitaper_to_smoothed_weights_" + str(latent_dim) + "_dimensions.h5")

    # Encode and decode some digits
    # # Note that we take them from the *test* set
    # encoded = encoder.predict(X_test)
    # decoded = decoder.predict(encoded)

    #_, accuracy = autoencoder.evaluate(X_train, y_train)
    #print(accuracy * 100)

    print("done")

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from scipy.signal import savgol_filter


def my_filter(x):
    return savgol_filter(x, 13, 1)

def try_smoothing_function(multitaper, smoothed, states, smoothing_function):

    # fit = SimpleExpSmoothing(multitaper, initialization_method="heuristic").fit(smoothing_level=0.2, optimized=False)
    multitaper = np.array(multitaper)
    smoothed = np.array(smoothed)
    training_data_states = np.array([s[0] for s in np.array(states)])
    smoothed_by_function = []

    for i in range(0, len(multitaper)):
        epoch = smoothing_function(multitaper[i]) #SimpleExpSmoothing(multitaper[i], initialization_method="heuristic").fit(smoothing_level=0.05, optimized=False).fittedvalues
        smoothed_by_function.append(epoch)
        #fit = SimpleExpSmoothing(multitaper[i], initialization_method="heuristic").fit(smoothing_level=0.05, optimized=False)

        #fit = SimpleExpSmoothing(multitaper[i], initialization_method="estimated").fit()


    n = 10  # How many digits we will display
    plt.figure(figsize=(30, 6))
    for i in range(n):
        idx = i + 20000

        # Display original
        ax = plt.subplot(3, n, i + 1)
        plt.plot(multitaper[i])
        plt.gray()

        # Display original
        ax = plt.subplot(3, n, i + 1 + n)
        plt.plot(np.array(smoothed_by_function[i]))
        plt.gray()

        # Display reconstruction
        ax = plt.subplot(3, n, i + 1 + (2 * n))
        plt.plot(smoothed[i])
        plt.gray()

    plt.show()

    lda = LDA(n_components=3)
    lda_encoded_data = lda.fit_transform(smoothed, training_data_states)
    plot_transformation = True
    if plot_transformation:
        fig = plt.figure()
        ax = Axes3D(fig)
        d = {0: 'blue',  # lm
             1: 'yellow',  # sws
             2: 'green',  # hmw
             3: 'red',  # rem
             }
        colors = [d[c] for c in np.array(training_data_states)]
        ax.scatter(lda_encoded_data[:, 0], lda_encoded_data[:, 1], lda_encoded_data[:, 2], c=colors, alpha=0.1, s=8)
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
        ax.set_zlabel('component 3')
        plt.show()

    print("done")
    return smoothed_by_function

def get_average_of_classes(multitaper, states):
    states = np.array(states)
    cluster_indexes = [np.where(states[:, 1] == "LMwake"), np.where(states[:, 1] == "SWS"),
                       np.where(states[:, 1] == "HMwake"), np.where(states[:, 1] == "REM")]
    averages = []
    for i in range(len(cluster_indexes)):
        class_examples = np.array(multitaper)[cluster_indexes[i]]
        averages.append(np.mean(class_examples, axis=0))

    plt.figure(figsize=(10, 6))
    n = len(averages)
    for i in range(n):
        idx = i + 20000

        # Display original
        ax = plt.subplot(1, n, i + 1)
        plt.plot(averages[i])
        plt.gray()

    plt.show()

    return averages

from statsmodels.tsa.arima.model import ARIMA

def do_nothing(x):
    return x

def arima(x):
    model = ARIMA(x, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit.fittedvalues

def moving_average(x):
    return np.convolve(x, np.ones(15)/15, mode='valid')

def ses_smoothing(x):
    #return SimpleExpSmoothing(x, initialization_method="estimated").fit().fittedvalues
    return SimpleExpSmoothing(x, initialization_method="heuristic").fit(smoothing_level=0.05, optimized=False).fittedvalues

def savgol_smoothing(x):
    savgol_smoothed_segment = x
    for iteration in range(4):
        savgol_smoothed_segment = my_filter(savgol_smoothed_segment)
    return savgol_smoothed_segment

def get_multitaper_data():
    mh = Offline_analysis.run_offline_analysis()
    return {
                "multitaper": mh.multitaper_df,
                "unsmoothed": mh.Sxx_norm_unsmoothed,
                 "smoothed": mh.Sxx_norm
            }

def smooth_data(x, smoothing_function):
    x = np.array(x)
    smoothed_by_function = []
    for i in range(0, len(x)):
        epoch = smoothing_function(x[i])  # SimpleExpSmoothing(multitaper[i], initialization_method="heuristic").fit(smoothing_level=0.05, optimized=False).fittedvalues
        smoothed_by_function.append(epoch)
    return smoothed_by_function

def compare_against_averages(x, states, averages):
    x = np.array(x)
    states = np.array([s[0] for s in np.array(states)])
    new_states = []
    for i in range(0, len(x)):
        fit_numbers = []
        for j in range(0, len(averages)):
            fit_numbers.append([np.linalg.norm(x[i] - averages[j])])
        min_item = np.min(fit_numbers)
        min_index = fit_numbers.index(min_item)
        new_states.append(min_index)
    return (len([i for i in range(len(states)) if states[i] == new_states[i]])/len(states)) * 100

multitaper_data_path = Config.base_path + "/vae_smoothing_trail_mouse_20210612.pkl"
data = Storage.load_or_recreate_file(multitaper_data_path, get_multitaper_data, False)
multitaper = data["multitaper"]
unsmoothed = data["unsmoothed"]
smoothed = data["smoothed"]
#run_keras_var_autoencoder(load_weights=False, mouse_data=multitaper)
states = Storage.load_from_file("C:/Users/bitsik0000/SleepData/210409/Ephys/states_210409_210409_B6J_m1.pkl")
averages = get_average_of_classes(smoothed, states)
difference = compare_against_averages(smoothed, states, averages)
print(difference)
smoothed_by_function = smooth_data(smoothed, ses_smoothing)
transformed = try_smoothing_function(unsmoothed, smoothed_by_function, states, do_nothing)
_ = get_average_of_classes(transformed, states)
#run_keras_basic_autencoder(smoothed, states, False)

#try_output_smoothed(unsmoothed, smoothed, states)
#try_predict_class(unsmoothed, states)
# check_labelled_data_on_autoencoder(multitaper_df, states)
