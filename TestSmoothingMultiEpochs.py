from smrx_version import Offline_analysis
import Storage
import numpy as np
from Common import plot_transition_states, train_lda
import Config
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from scipy.signal import savgol_filter
from tsmoothie.smoother import LowessSmoother, KalmanSmoother, SplineSmoother, GaussianSmoother
from tsmoothie.smoother import BinnerSmoother, ExponentialSmoother, ConvolutionSmoother, DecomposeSmoother, SpectralSmoother

if __name__ == '__main__':

    def savgol_filter(x):
        return savgol_filter(x, 13, 1)

    def try_smoothing_function_across_epochs(to_transform, pre_smoothed, states, smoothing_function):

        epoch_size = to_transform.shape[1]
        original_number_of_epochs = len(to_transform)
        to_transform = np.array(to_transform)
        pre_smoothed = np.array(pre_smoothed)
        numeric_states = np.array([s[0] for s in np.array(states)])

        to_transform_flattened = np.array(to_transform).flatten()
        whole_series_smoothed = smoothing_function(to_transform_flattened)
        number_of_epochs = int(len(whole_series_smoothed)/epoch_size)
        epochs_to_drop = original_number_of_epochs - number_of_epochs
        smoothed = whole_series_smoothed.reshape(number_of_epochs, epoch_size)
        numeric_states = numeric_states[epochs_to_drop:]

        train_lda(smoothed, numeric_states, True)

        plot_transition_states(to_transform, smoothed, pre_smoothed, numeric_states)



    def do_nothing(x):
        return x

    def binner_method(x):
        smoother = BinnerSmoother(n_knots=8040)
        transformed = smoother.smooth(x)
        return np.array(transformed.smooth_data[0])

    def spectral_method(x):
        smoother = SpectralSmoother(smooth_fraction=0.2, pad_len=2000)
        transformed = smoother.smooth(x)
        return np.array(transformed.smooth_data[0])

    def convolution_method(x):
        smoother = ConvolutionSmoother(window_len=1, window_type='blackman')
        transformed = smoother.smooth(x)
        return np.array(transformed.smooth_data[0])

    def spline_method(x):
        smoother = SplineSmoother(spline_type='natural_cubic_spline', n_knots=50)
        transformed = smoother.smooth(x)
        return np.array(transformed.smooth_data[0])

    def kalman_method(x):
        smoother = KalmanSmoother(component='level', component_noise={'level': 0.1})
        transformed = smoother.smooth(x)
        return np.array(transformed.smooth_data[0])

    def exponential_method(x):
        smoother = ExponentialSmoother(window_len=1407, alpha=0.5)
        transformed = smoother.smooth(x)
        return np.array(transformed.smooth_data[0])


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
            savgol_smoothed_segment = savgol_filter(savgol_smoothed_segment)
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

    from tsmoothie.smoother import LowessSmoother
    def loewess_method(x):
        smoother = LowessSmoother(smooth_fraction=0.1, iterations=1)
        transformed = smoother.smooth(x)
        return transformed


    multitaper_data_path = Config.base_path + "/vae_smoothing_trail_mouse_20210612.pkl"
    data = Storage.load_or_recreate_file(multitaper_data_path, get_multitaper_data, False)
    multitaper = data["multitaper"]
    unsmoothed = data["unsmoothed"]
    smoothed = data["smoothed"]
    states = Storage.load_from_file("C:/Users/matthew.grant/source/repos/ClosedLoopEEG/data/Ephys/states_210409_210409_B6J_m1.pkl")

    transformed = try_smoothing_function_across_epochs(unsmoothed, smoothed, states, binner_method)


