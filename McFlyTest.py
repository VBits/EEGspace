from smrx_version import Offline_analysis
import Storage
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import Config
from keras.utils.np_utils import to_categorical
import mcfly
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def try_predict_class_decisiontree(series, states):

    states = np.array(states)
    training_data_states = np.array([s[0] for s in states])

    X = np.array(series)
    y = np.array(training_data_states)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))

def try_predict_class(multitaper, states, load_weights=False):
    cluster_indexes = [np.where(states[:] == 0), np.where(states[:] == 1),
                       np.where(states[:] == 2), np.where(states[:] == 3)]
    weights = {i:len(cluster_indexes[i][0])/len(states) for i in range(4)}
    states = np.array(states)
    training_data_states = np.array([s[0] for s in states])
    multitaper = np.array(multitaper)
    X = np.array(multitaper)
    y = np.array(training_data_states)
    #rnd.seed(42)

    latent_dim = 3

    num_classes = len(np.unique(training_data_states))

    X = np.array(multitaper)
    y = np.array(training_data_states)
    y = to_categorical(y, num_classes=4)

    X = X.astype('float32')
    y = y.astype('float32')
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # X = X.reshape((len(X), np.prod(X.shape[1:])))
    # y = y.reshape((len(y), np.prod(y.shape[1:])))
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # y_train = np.expand_dims(y_train, axis=1).astype(int)
    # y_test = np.expand_dims(y_test, axis=1).astype(int)

    best_model, best_params, best_model_type, knn_acc = mcfly.find_architecture.find_best_architecture(X_train, y_train, X_test, y_test,
                       verbose=True, number_of_models=25, nr_epochs=5, subset_size=25000, class_weight=weights)

    best_model.save(Config.base_path + "/unsmoothed_best_from_mcfly.h5")
    print(best_params)
    print(best_model_type)
    history = best_model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=100,
                        validation_data=(X_test, y_test))
    print(history.history)
    best_model.save(Config.base_path + "/retrained_unsmoothed_best_from_mcfly.h5")
    print("done")


def try_mcfly_model(multitaper, states):
    cluster_indexes = [np.where(states[:] == 0), np.where(states[:] == 1),
                       np.where(states[:] == 2), np.where(states[:] == 3)]
    weights = {i:len(cluster_indexes[i][0])/len(states) for i in range(4)}
    states = np.array(states)
    training_data_states = np.array([s[0] for s in states])
    multitaper = np.array(multitaper)

    #rnd.seed(42)

    latent_dim = 3

    num_classes = len(np.unique(training_data_states))

    X = np.array(multitaper)
    y = np.array(training_data_states)
    y = to_categorical(y, num_classes=4)

    X = X.astype('float32')
    y = y.astype('float32')
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # X = X.reshape((len(X), np.prod(X.shape[1:])))
    # y = y.reshape((len(y), np.prod(y.shape[1:])))
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # y_train = np.expand_dims(y_train, axis=1).astype(int)
    # y_test = np.expand_dims(y_test, axis=1).astype(int)

    model = mcfly.modelgen .generate_resnet_model(X_train.shape, 4, 99, 31, network_depth=2,
                    learning_rate=0.00024299650557849448, regularization_rate=0.0018542104762003598, metrics=['accuracy'])

    history = model.fit(X_train, y_train,
            epochs=100,
            batch_size=256,
            validation_data=(X_test, y))
    model.save(Config.base_path + "/resnet_from_mcfly.h5")
    print(history.history)

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
from sklearn.decomposition import PCA

def check_max_3_subpoints(series, states):

    series = np.array(series)
    encoded = []

    for i in range(len(series)):
        epoch = series[i]
        point_1 = np.array(epoch[0:30]).max()
        point_2 = np.array(epoch[30:80]).max()
        point_3 = np.array(epoch[80:201]).max()
        #point_4 = np.array(epoch[120:200]).max()
        encoded.append([point_1, point_2, point_3])#, point_4])

    encoded = np.array(encoded)
    # pca = PCA(n_components=3)
    # encoded = pca.fit_transform(encoded)

    d = {'HMwake': 'blue',
         'LMwake': 'green',
         'SWS': 'yellow',
         'REM': 'red',
         }

    fig = plt.figure()
    ax = Axes3D(fig)
    n = 40000
    rand_idx = np.random.choice(len(encoded), size=n, replace=False)
    subset = encoded#[rand_idx]
    colors = [d[c] for c in np.array(states)[:, 1]]#[rand_idx]
    ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c=colors, alpha=0.1, s=8)
    ax.set_xlabel('ld1')
    ax.set_ylabel('ld2')
    ax.set_zlabel('ld3')
    plt.show()

    print("done")

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
                "eeg": mh.EEG_data,
                "fs": mh.EEG_fs,
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

def smooth_data(x, smoothing_function):
    x = np.array(x)
    smoothed_by_function = []
    for i in range(0, len(x)):
        epoch = smoothing_function(x[i])  # SimpleExpSmoothing(multitaper[i], initialization_method="heuristic").fit(smoothing_level=0.05, optimized=False).fittedvalues
        smoothed_by_function.append(epoch)
    return smoothed_by_function

def ses_smoothing(x):
    #return SimpleExpSmoothing(x, initialization_method="estimated").fit().fittedvalues
    return SimpleExpSmoothing(x, initialization_method="heuristic").fit(smoothing_level=0.05, optimized=False).fittedvalues

from sklearn.metrics import confusion_matrix
from sklearn import svm

def try_predict_class_SVM(series, states):

    states = np.array(states)
    training_data_states = np.array([s[0] for s in states])

    X = np.array(series)
    y = np.array(training_data_states)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
    rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)
    accuracy_lin = linear.score(X_test, y_test)
    accuracy_poly = poly.score(X_test, y_test)
    accuracy_rbf = rbf.score(X_test, y_test)
    accuracy_sig = sig.score(X_test, y_test)
    print("Accuracy     Linear    Kernel:", accuracy_lin)
    print("Accuracy    Polynomial    Kernel:", accuracy_poly)
    print("Accuracy    Radial    Basis    Kernel:", accuracy_rbf)
    print("Accuracy    Sigmoid    Kernel:", accuracy_sig)


multitaper_data_path = Config.base_path + "/mcfly_test_data.pkl"
data = Storage.load_or_recreate_file(multitaper_data_path, get_multitaper_data, False)
eeg = data["eeg"]
fs = data["fs"]
multitaper = data["multitaper"]
unsmoothed = data["unsmoothed"]
smoothed = data["smoothed"]
#run_keras_var_autoencoder(load_weights=False, mouse_data=multitaper)
states = Storage.load_from_file("C:/Users/matthew.grant/source/repos/ClosedLoopEEG/data/Ephys/states_210409_210409_B6J_m1.pkl")
# averages = get_average_of_classes(smoothed, states)
# difference = compare_against_averages(smoothed, states, averages)
# print(difference)
# smoothed_by_function = smooth_data(smoothed, ses_smoothing)
# transformed = try_smoothing_function(unsmoothed, smoothed_by_function, states, do_nothing)
# _ = get_average_of_classes(transformed, states)
#run_keras_basic_autencoder(smoothed, states, False)

#try_output_smoothed(unsmoothed, smoothed, states)
#MCFLY
#try_mcfly_model(unsmoothed, states)
epoch = fs*2
max_epochs= int(len(eeg)/epoch)
max_length = int(max_epochs*epoch)
split_eeg = np.array(np.split(eeg[:max_length], max_epochs))
print(split_eeg.shape)



#transformed = smooth_data(unsmoothed, ses_smoothing)
#try_predict_class(split_eeg[:len(states)], states)
#try_predict_class_decisiontree(multitaper, states)
try_predict_class_SVM(multitaper, states)


#check_max_3_subpoints(unsmoothed, states)
# check_labelled_data_on_autoencoder(multitaper_df, states)
