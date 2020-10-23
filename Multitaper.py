import nitime.algorithms as tsa
from scipy.signal import detrend
from utils import *

mouse_num = 3
mouse_object_path = Config.base_path + "/mouse_object_" + str(mouse_num)
multitaper_dataframe_path = Config.base_path + "/multitaper_mouse_" + str(mouse_num) + "_without_norm.pkl"
multitaper_data_path = Config.base_path + "/multitaper_mouse_" + str(mouse_num) + ".pkl"

do_multitaper = True
plot_multitaper = True

if Config.generate_mouse_object:
    create_mouse_object(mouse_num, mouse_object_path)

if do_multitaper:
    f = open(mouse_object_path, 'rb')
    mh = pickle.load(f)

    # Set widnow, overlap and calculate multitaper
    window_length = 4 * int(mh.EEG_fs)
    window_step = 2 * int(mh.EEG_fs)

    EEG_data = mh.EEG_data

    window_starts = np.arange(0, len(EEG_data) - window_length + 1, window_step)
    EEG_segs = detrend(EEG_data[list(map(lambda x: np.arange(x, x + window_length), window_starts))])
    freqs, psd_est, var_or_nu = tsa.multi_taper_psd(EEG_segs, Fs=mh.EEG_fs,NW=4, adaptive=False, jackknife=False, low_bias=True)  # , dpss=dpss, eigvals=eigvals)
    multitaper_df = pd.DataFrame(index=freqs,data=psd_est.T)

    f = open(multitaper_dataframe_path, 'wb')
    pickle.dump(multitaper_df, f)

    def SG_filter(x):
        return scipy.signal.savgol_filter(x, 41, 2)

    # Log scale
    Sxx = 10 * np.log(multitaper_df.values)

    Sxx_df = pd.DataFrame(data=Sxx, index=mh.f)

    # horizontal axis (time)
    iterations = 4
    for i in range(iterations):
        Sxx_df = Sxx_df.apply(SG_filter, axis=1, result_type='expand')

    def density_calc(dataframe, boundary=(-100, 90)):
        # now calculate the bins for each frequency
        density_mat = []
        mean_density = []
        for i in range(len(dataframe.index)):
            density, bins = np.histogram(dataframe.iloc[i, :], bins=5000, range=boundary, density=True)

            density_mat.append(density)
            mean_density.append(dataframe.iloc[i, :].mean())

        density_mat = np.array(density_mat)
        bins = (bins[1:] + bins[:-1]) / 2
        return density_mat, bins


    density_mat, bins = density_calc(Sxx_df, boundary=(-100, 90))  # -1,1550

    density_df = pd.DataFrame(index=bins, data=density_mat.T, columns=mh.f)
    for i in range(iterations):
        density_df = density_df.apply(SG_filter, axis=0, result_type='expand')

    baseline = np.argmax(density_df.values > 0.01, axis=0)

    norm = 0 - bins[baseline]
    Sxx_df_norm = Sxx_df.add(norm, axis=0)

    multitaper_path = Config.base_path + "/multitaper_mouse_" + str(mouse_num)

    f = open(multitaper_data_path, 'wb')
    pickle.dump(Sxx_df_norm, f)

if plot_multitaper:
    f = open(multitaper_dataframe_path, 'rb')
    Sxx_df = pickle.load(f)
    n = 10
    s = Sxx_df
    plt.figure(figsize=(20, 4))
    offset = 20000
    for i, d in enumerate(np.array(s.T[offset:offset+n])):
        ax = plt.subplot(1, n, i + 1)
        plt.plot(d)
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    print("done")
