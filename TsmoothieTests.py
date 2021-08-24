from smrx_version import Offline_analysis
import Storage
import numpy as np
from tsmoothie.smoother import LowessSmoother, KalmanSmoother, SplineSmoother, GaussianSmoother
from tsmoothie.smoother import BinnerSmoother, ExponentialSmoother, ConvolutionSmoother, DecomposeSmoother, SpectralSmoother
import matplotlib.pyplot as plt
import Config
from mpl_toolkits.mplot3d import Axes3D
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

if __name__ == '__main__':

    from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
    from scipy.signal import savgol_filter


    def my_filter(x):
        return savgol_filter(x, 13, 1)

    def try_smoothing_function_on_subset(subset_size, multitaper, smoothed, smoothing_function):

        # fit = SimpleExpSmoothing(multitaper, initialization_method="heuristic").fit(smoothing_level=0.2, optimized=False)
        multitaper = np.array(multitaper[:subset_size])
        smoothed = np.array(smoothed[:subset_size])
        smoothed_by_function = []

        for i in range(0, len(multitaper)):
            epoch = smoothing_function(multitaper[i])
            smoothed_by_function.append(epoch)

        smoothed_by_function = np.array(smoothed_by_function)

        plt.figure(figsize=(subset_size * 3, 6))
        for i in range(subset_size):

            ax = plt.subplot(3, subset_size, i + 1)
            ax.get_xaxis().set_visible(False)
            plt.plot(multitaper[i])
            plt.gray()

            ax = plt.subplot(3, subset_size, i + 1 + subset_size)
            ax.get_xaxis().set_visible(False)
            plt.plot(np.array(smoothed_by_function[i]))
            plt.gray()

            ax = plt.subplot(3, subset_size, i + 1 + (2 * subset_size))
            plt.plot(smoothed[i])
            plt.gray()

        plt.show()
        print("done")

    def try_smoothing_function(original, multitaper, smoothed, states, smoothing_function):

        # fit = SimpleExpSmoothing(multitaper, initialization_method="heuristic").fit(smoothing_level=0.2, optimized=False)
        original = np.array(original)
        multitaper = np.array(multitaper)
        smoothed = np.array(smoothed)
        training_data_states = np.array([s[0] for s in np.array(states)])
        smoothed_by_function = []

        for i in range(0, len(multitaper)):
            epoch = smoothing_function(multitaper[i]) #SimpleExpSmoothing(multitaper[i], initialization_method="heuristic").fit(smoothing_level=0.05, optimized=False).fittedvalues
            smoothed_by_function.append(epoch)
            #fit = SimpleExpSmoothing(multitaper[i], initialization_method="heuristic").fit(smoothing_level=0.05, optimized=False)

            #fit = SimpleExpSmoothing(multitaper[i], initialization_method="estimated").fit()
        smoothed_by_function = np.array(smoothed_by_function)

        averages = get_average_of_classes(smoothed_by_function, states)

        transitions = []
        for i in range(len(states)):
            if (i-5) > 0 and (i + 1) < (len(states)-5) and training_data_states[i] != training_data_states[i+1]:
                transitions.append(i)
        transitions = np.array(transitions)

        n = 10
        for num in range(n):
            transition_index = transitions[num]
            plt.figure(figsize=(30, 6))
            for i in range(n):
                new_i = transition_index-5 + i
                # Display original
                ax = plt.subplot(3, n, i + 1)
                #ax.title.set_text("multitaper applied, state: " + str(training_data_states[new_i]))
                ax.get_xaxis().set_visible(False)
                plt.plot(multitaper[new_i])
                plt.gray()

                # Display original
                ax = plt.subplot(3, n, i + 1 + n)
                #ax.title.set_text("log applied, state: " + str(training_data_states[new_i]))
                ax.get_xaxis().set_visible(False)
                plt.plot(np.array(smoothed_by_function[new_i]))
                plt.gray()

                # Display reconstruction
                ax = plt.subplot(3, n, i + 1 + (2 * n))
                #ax.title.set_text("savgol applied, state: " + str(training_data_states[new_i]))
                plt.plot(smoothed[new_i])
                plt.gray()

        plt.show()

        lda = LDA(n_components=3)
        lda_encoded_data = lda.fit_transform(smoothed_by_function, training_data_states)
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
        return SimpleExpSmoothing(x, initialization_method="heuristic").fit(smoothing_level=0.5, optimized=False).fittedvalues

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

    def loewess_method(x):
        smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
        transformed = smoother.smooth(x)
        return np.array(transformed.smooth_data[0])

    def kalman_method(x):
        smoother = KalmanSmoother(component='level', component_noise={'level':0.1})
        transformed = smoother.smooth(x)
        return np.array(transformed.smooth_data[0])

    def gaussian_method(series):
        from tsmoothie.smoother import GaussianSmoother
        smoother = GaussianSmoother(n_knots=15, sigma=0.1)
        smoothed_series = np.array(smoother.smooth(series).smooth_data[0])
        return smoothed_series

    def spectral_method(x):
        smoother = SpectralSmoother(smooth_fraction=0.1, pad_len=75)
        transformed = smoother.smooth(x)
        return np.array(transformed.smooth_data[0])

    def spline_method(x):
        smoother = SplineSmoother(spline_type='natural_cubic_spline', n_knots=2000)
        transformed = smoother.smooth(x)
        return np.array(transformed.smooth_data[0])

    # def decompose_method(x):
    #     smoother = DecomposeSmoother(smooth_type='convolution')
    #     transformed = smoother.smooth(x)
    #     return np.array(transformed.smooth_data[0])

    multitaper_data_path = Config.base_path + "/vae_smoothing_trail_mouse_20210612.pkl"
    data = Storage.load_or_recreate_file(multitaper_data_path, get_multitaper_data, False)
    multitaper = data["multitaper"]
    unsmoothed = data["unsmoothed"]
    smoothed = data["smoothed"]
    states = Storage.load_from_file("C:/Users/matthew.grant/source/repos/ClosedLoopEEG/data/Ephys/states_210409_210409_B6J_m1.pkl")
    method = spline_method
    series = unsmoothed
    transformed = try_smoothing_function_on_subset(10, series, smoothed, method)
    transformed = try_smoothing_function(multitaper, series, smoothed, states, method)

