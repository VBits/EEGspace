from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from scipy.signal import savgol_filter
import Config
import Storage
from smrx_version import Offline_analysis


def get_state_mapping_from_list(states_list):
    return list(set([(x[0], x[1]) for x in states_list.tolist()]))


def train_lda(series, numerical_states, show_plot=False):
    lda = LDA(n_components=3)
    lda_encoded_data = lda.fit_transform(series, numerical_states)
    if show_plot:
        plot_lda(lda_encoded_data, numerical_states)
    return lda, lda_encoded_data


def get_multitaper_data():
    mh = Offline_analysis.run_offline_analysis()
    return {
                "multitaper": mh.multitaper_df,
                "unsmoothed": mh.Sxx_norm_unsmoothed,
                 "smoothed": mh.Sxx_norm
            }

def plot_lda(lda_encoded_data, states_numeric):
   # plot state predictions
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(lda_encoded_data[:, 0], lda_encoded_data[:, 1], lda_encoded_data[:, 2], c=states_numeric,
               alpha=0.2, s=4, cmap='Accent')
    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_zlabel('C3')
    plt.title('LDA')
    # plt.savefig(figureFolder+'LDA ANN clusters' + figure_tail, dpi=dpi)
    plt.show()


def load_offline_data():
    multitaper_data_path = Config.base_path + "/vae_smoothing_trail_mouse_20210612.pkl"
    data = Storage.load_or_recreate_file(multitaper_data_path, get_multitaper_data, False)
    multitaper = data["multitaper"]
    unsmoothed = data["unsmoothed"]
    smoothed = data["smoothed"]
    states = Storage.load_from_file("C:/Users/matthew.grant/source/repos/ClosedLoopEEG/data/Ephys/states_210409_210409_B6J_m1.pkl")
    return multitaper, unsmoothed, smoothed, states


def states_to_numeric_version(states):
    return np.array([s[0] for s in np.array(states)])


def apply_savgol(series, window_size=41, iterations=10):
    for i in range(0, iterations):
        series = savgol_filter(series, window_size, 1)
    return series


def plot_transition_states(series1, series2, series3, numeric_states, series1_label='Series 1',
                           series2_label='Series 2', series3_label='Series 3'):
    transitions = []
    for i in range(len(numeric_states)):
        if (i - 5) > 0 and (i + 1) < (len(numeric_states) - 5) and numeric_states[i] != numeric_states[i + 1]:
            transitions.append(i)
    transitions = np.array(transitions)

    n = 10
    for num in range(n):
        transition_index = transitions[num]
        plt.figure(figsize=(30, 6))
        for i in range(n):
            new_i = transition_index - 5 + i
            # Display original
            ax = plt.subplot(3, n, i + 1)
            ax.title.set_text(series1_label + ": " + str(numeric_states[new_i]))
            ax.get_xaxis().set_visible(False)
            plt.plot(series1[new_i])
            plt.gray()

            # Display original
            ax = plt.subplot(3, n, i + 1 + n)
            ax.title.set_text(series2_label + ": " + str(numeric_states[new_i]))
            ax.get_xaxis().set_visible(False)
            plt.plot(np.array(series2[new_i]))
            plt.gray()

            # Display reconstruction
            ax = plt.subplot(3, n, i + 1 + (2 * n))
            ax.title.set_text(series3_label + ": " + str(numeric_states[new_i]))
            plt.plot(series3[new_i])
            plt.gray()

    plt.show()


def transition_state_misclassification_percentage(original_numeric_states, predicted_numerical_states):
    transitions = []
    for i in range(len(original_numeric_states)):
        if (i - 5) > 0 and (i + 1) < (len(original_numeric_states) - 5) and original_numeric_states[i] != original_numeric_states[i + 1]:
            transitions.append(i)
    transitions = np.array(transitions)

    predicted_numerical_states[transitions]
