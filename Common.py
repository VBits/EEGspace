from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from scipy.signal import savgol_filter
import Config
import Storage
from smrx_version import Offline_analysis
from sklearn.metrics import confusion_matrix


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
               alpha=0.3, s=5, cmap='Accent')
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
    states = Storage.load_from_file(Config.base_path + "/data/Ephys/states_210409_210409_B6J_m1.pkl")
    return multitaper, unsmoothed, smoothed, states


def states_to_numeric_version(states):
    return np.array([s[0] for s in np.array(states)])


def apply_savgol(series, window_size=41, iterations=10):
    for i in range(0, iterations):
        series = savgol_filter(series, window_size, 1)
    return series


def plot_data_overlayed(data, plot_title="plot", labels=None, colors=None, weights=None):
    for i in range(len(data)):#subplots
        ax = plt.subplot(1, len(data), i + 1)
        for j in range(len(data[i])):#overlapping plots
            if labels is None:
                ax.plot(data)
            else:
                color = colors[j] if colors is not None else '-b'
                weight = weights[j] if weights is not None else 1
                ax.plot(data[i][j], color, label=labels[j], linewidth=weight)
        plt.legend()
        if plot_title is not None:
            plt.title(plot_title)
    plt.show()


def select_random_data(series, size):
    rand_idx = np.random.choice(series.shape[0], size=size, replace=False)
    return np.array(series)[rand_idx]


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


def compare_states_for_accuracy(original_numeric_states, predicted_numerical_states):
    accurate_predictions = np.where(original_numeric_states == predicted_numerical_states)
    percentage = (len(accurate_predictions[0]) / original_numeric_states.shape[0]) * 100
    return percentage


def get_transition_indexes(original_numeric_states):
    transitions = []
    for i in range(len(original_numeric_states)):
        if (i - 5) > 0 and (i + 1) < (len(original_numeric_states) - 5) and original_numeric_states[i] != original_numeric_states[i + 1]:
            transitions.append([i-1, i, i+1, i+2, i+3, i+4])
    transitions = np.array(transitions)
    return transitions


def transition_state_misclassification_percentages(original_numeric_states, predicted_numerical_states):
    transitions = get_transition_indexes(original_numeric_states)
    percentages = []
    for i in range(0, len(transitions[0])):
        original_transitions = np.array(original_numeric_states[transitions[:, i]])
        predicted_transitions = np.array(predicted_numerical_states[transitions[:, i]])
        accurate_predictions = np.where(original_transitions == predicted_transitions)
        percentages.append((len(accurate_predictions[0])/transitions.shape[0]) * 100)
    return percentages


def print_confusion_matrix(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))


def graph_transition_averages(series, states):
    states_numeric = states_to_numeric_version(states)
    state_mappings = get_state_mapping_from_list(np.array(states))
    transitions = get_transition_indexes(states_numeric)
    unique_states = list(set(states_numeric))
    transition_averages = []
    all_state_average = []
    series = np.array(series)
    #for i in range(transitions.shape[1]):
    current_offset = transitions[:, 2]
    for i in range(len(unique_states)):
        transition_state_indexes = [index for index in current_offset if states_numeric[index] == unique_states[i]]
        all_state_indexes = [index for index, _ in enumerate(states_numeric) if states_numeric[index] == unique_states[i]]
        transition_average = np.mean(series[transition_state_indexes], axis=0)
        transition_averages.append(transition_average)
        all_average = np.mean(series[all_state_indexes], axis=0)
        all_state_average.append(all_average)

    for i in range(len(transition_averages)):
        state_name = [mapping[1] for mapping in state_mappings if unique_states[i] == mapping[0]][0]
        ax = plt.subplot(1, len(transition_averages), i + 1)
        plt.plot(transition_averages[i], "-b", label="transition state averages")
        plt.plot(all_state_average[i], "-r", label="all state averages")
        plt.legend(loc="upper left")
        plt.title(state_name)
    plt.show()
