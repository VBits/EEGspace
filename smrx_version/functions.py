from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
import pandas as pd
import h5py
import scipy
import scipy.signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import Normalizer
from scipy.spatial import cKDTree
from scipy.signal import decimate, butter, dlti
import inspect
import nitime.algorithms as tsa
from scipy.signal import detrend
import datetime
import os
from math import floor
from sonpy import lib as sp


def inspect_function(f):
    code, line_no = inspect.getsourcelines(f)
    print(''.join(code))

# The mouse class object ()
class Mouse:
    def __init__(self, genotype, pos):
        self.genotype = genotype  # instance variable unique to each instance
        self.pos = pos
        self.colors ={'Wake': '#80a035',  # green
                      'HMwake':'#80a035',
                     'LMwake':'#617D21',
                  'Sleep': '#353377',  # blue
                  'SWS': '#353377',  # blue
                  'REM': '#aa6339',  # orange
                  'ambiguous': '#ff0000',  # red
                  }

    def __repr__(self):
        return "Mouse in position {}, genotype {}".format(self.pos, self.genotype)

    def add_data(self, Folder, FileMat):
        self.f = h5py.File(Folder + FileMat,'r')
        self.Ch_name = list(self.f.keys())
        self.Mouse_Ch = [s for s in self.Ch_name if "G{}".format(self.pos) in s]
        self.EEG_data = self.f["{}".format(self.Mouse_Ch[0])]["values"][0, :]

        start = pd.DataFrame(self.f['file']['start'][0].reshape(6,1).T, columns = ['year',
                              'month','day','hour','minute','second'])
        self.start = pd.to_datetime(start)
        self.interval = self.f["{}".format(self.Mouse_Ch[0])]['interval'][0][0]
        self.EEG_fs = 1 / self.f["{}".format(self.Mouse_Ch[0])]['interval'][0][0]
        if len(self.Mouse_Ch) == 2:
            self.EMG_data = self.f["{}".format(self.Mouse_Ch[1])]["values"][0, :]
            self.EMG_fs = 1 / self.f["{}".format(self.Mouse_Ch[1])]['interval'][0][0]

    def read_smrx(self,Filepath):
        # Get file path
        self.figure_tail = ' - {} - {}.png'.format(self.pos, self.genotype)
        self.FilePath = Filepath
        print('Loading Mouse {} from {}'.format(self.pos,self.FilePath))

        # Open file
        self.File = sp.SonFile(self.FilePath, True)

        if self.File.GetOpenError() != 0:
            print('Error opening file:', sp.GetErrorString(self.File.GetOpenError()))
            quit()
        WaveChan = self.pos - 1
        self.Ch_units = self.File.GetChannelUnits(WaveChan)
        self.Ch_name = self.File.GetChannelTitle(WaveChan)

        # Get number of seconds to read
        dMaxSeconds = self.File.ChannelMaxTime(WaveChan) * self.File.GetTimeBase()

        # Prepare for plotting
        dPeriod = self.File.ChannelDivide(WaveChan) * self.File.GetTimeBase()
        nPoints = floor(dMaxSeconds / dPeriod)

        self.EEG_data = np.array(self.File.ReadFloats(WaveChan, nPoints, 0))

        start = pd.DataFrame(np.reshape(self.File.GetTimeDate()[::-1][:-1], (1, 6)),
                             columns=['year', 'month', 'day', 'hour', 'minute', 'second'])
        self.start = pd.to_datetime(start)
        self.EEG_fs = 1 / dPeriod
        self.EEG_ideal_fs = self.File.GetIdealRate(WaveChan)

    def gen_folder(self,EphysDir,Folder,all_mice=None):
        date = datetime.datetime.now().strftime("%y%m%d")
        if all_mice is None:
            figureFolder = EphysDir + Folder + 'Mouse_{}_{}/'.format(self.pos, date)
        else:
            figureFolder = EphysDir + Folder + 'All_Mice_{}/'.format(date)
        if not os.path.exists(figureFolder):
            print('Directory created')
            os.makedirs(os.path.dirname(figureFolder), exist_ok=True)
        else:
            print('Directory exists')
        return figureFolder

    def downsample_EGG(self,target_fs=100):
        '''
        Downsample the data to a target frequency of 100Hz

        You can also replace the Butterworth filter with Bessel filter or the default Chebyshev filter.
        system = dlti(*bessel(4,0.99))
        system = dlti(*cheby1(3,0.05,0.99))
        All filters produced very similar results for downsampling from 200Hz to 100Hz
        '''
        self.EEG_fs = round(self.EEG_fs)
        rate = self.EEG_fs/ target_fs
        system = dlti(*butter(4,0.99))
        self.EEG_data = decimate(self.EEG_data, round(rate), ftype=system, zero_phase=True)
        self.EEG_fs = self.EEG_fs / rate


    def multitaper(self,resolution=2):
        '''
        :param resolution: specify the desired resolution in seconds
        :return:
        '''
        window_length = 2 * resolution * int(self.EEG_fs)
        window_step = resolution * int(self.EEG_fs)
        window_starts = np.arange(0, len(self.EEG_data) - window_length + 1, window_step)

        EEG_segs = detrend(self.EEG_data[list(map(lambda x: np.arange(x, x + window_length), window_starts))])

        freqs, psd_est, var_or_nu = tsa.multi_taper_psd(EEG_segs, Fs=self.EEG_fs, NW=4, adaptive=False, jackknife=False,
                                                        low_bias=True)  # , dpss=dpss, eigvals=eigvals)

        # self.multitaper_df = pd.DataFrame(index=freqs, data=psd_est.T)
        time_idx = pd.date_range(start=self.start[0], freq='{}ms'.format(window_step/self.EEG_fs*1000), periods=len(psd_est))
        self.multitaper_df = pd.DataFrame(index=time_idx, data=psd_est,columns=freqs)

    def process_spectrum(self,smooth_iter=4, window_size=41):
        ## Normalize the data and plot density spectrogram
        def SG_filter(x):
            return scipy.signal.savgol_filter(x, window_size, 2)

        # Log scale
        Sxx_df = 10 * np.log(self.multitaper_df.T)

        # horizontal axis (time)
        for i in range(smooth_iter):
            Sxx_df = Sxx_df.apply(SG_filter, axis=1, result_type='expand')

        self.Sxx_df = pd.DataFrame(data=Sxx_df.T.values, columns=self.multitaper_df.columns,
                                     index=self.multitaper_df.index)

        density_mat, bins = density_calc(Sxx_df, boundary=(-100, 90))  # -1,1550

        density_df = pd.DataFrame(index=bins, data=density_mat.T, columns=self.multitaper_df.columns)
        for i in range(smooth_iter):
            density_df = density_df.apply(SG_filter, axis=0, result_type='expand')

        self.baseline = np.argmax(density_df.values > 0.01, axis=0)

        self.norm = 0 - bins[self.baseline]
        Sxx_norm = Sxx_df.add(self.norm, axis=0)
        self.density_norm, self.power_bins = density_calc(Sxx_norm, boundary=(-25, 50))
        self.Sxx_norm = pd.DataFrame(data=Sxx_norm.T.values,columns=self.multitaper_df.columns,
                                     index=self.multitaper_df.index)

    def process_spectrum_detached(self,smooth_iter=4, window_size=41):
        ## Normalize the data and plot density spectrogram
        def SG_filter(x):
            return scipy.signal.savgol_filter(x, window_size, 2)

        # Log scale
        Sxx_df = 10 * np.log(self.multitaper_df.T)

        # horizontal axis (time)
        for i in range(smooth_iter):
            Sxx_df = Sxx_df.apply(SG_filter, axis=1, result_type='expand')

        density_mat, bins = density_calc(Sxx_df, boundary=(-100, 90))  # -1,1550

        density_df = pd.DataFrame(index=bins, data=density_mat.T, columns=self.multitaper_df.columns)
        for i in range(smooth_iter):
            density_df = density_df.apply(SG_filter, axis=0, result_type='expand')

        baseline = np.argmax(density_df.values > 0.01, axis=0)

        norm = 0 - bins[baseline]
        Sxx_norm = Sxx_df.add(norm, axis=0)
        density_norm, power_bins = density_calc(Sxx_norm, boundary=(-25, 50))
        Sxx_norm = pd.DataFrame(data=Sxx_norm.T.values,columns=self.multitaper_df.columns,
                                     index=self.multitaper_df.index)
        return Sxx_norm


    def PCA(self, window_size = 11,normalizer=False,robust =False):
        if self.LP_filter:
            if normalizer:
                self.x = Normalizer().fit_transform(self.df)
                print('Using Normalizer')
            elif robust:
                self.x = RobustScaler(quantile_range=(1, 99)).fit_transform(self.df)
                print('Using Robust Scaler')
            else:
                self.x = StandardScaler().fit_transform(self.df)
                print('Using Standard Scaler')
        else:
            self.x = StandardScaler().fit_transform(self.df.rolling(window_size, center=True,
                                                     win_type=None, min_periods=2).mean())
            print('Using Standard Scaler and rolling mean')
        print('LP filter was set to {}'.format(self.LP_filter))
        pca = PCA(n_components=2)
        self.pC = pca.fit_transform(self.x)
        self.pC_df = pd.DataFrame(data=self.pC,columns=['pC1','pC2'],index=self.df.index)

    def knn_pred(self, clf, transform='PCA'):
        # predict in 2D
        self.state_df = pd.DataFrame(index=self.Sxx_norm.index)
        if transform =='PCA':
            print('Predicting clusters using PCA')
            self.state_df['clusters_knn'] = clf.predict(self.pC_df)
        elif transform =='LDA':
            print('Predicting clusters using LDA')
            self.state_df['clusters_knn'] = clf.predict(self.LD_df)
        else:
            print('Cannot recognise PCA or LDA data')

        Nclusters = len(self.state_df['clusters_knn'].unique())

        # Count state instances after finding which code has higher average T_D.
        # Descending order(REM, Wake, SWS)
        state_code = np.zeros(Nclusters)
        for i in range(Nclusters):
            delta = self.Sxx_norm.loc[:, 1:4][self.state_df['clusters_knn'] == i].mean().mean()
            theta = self.Sxx_norm.loc[:, 7:10][self.state_df['clusters_knn'] == i].mean().mean()
            gamma = self.Sxx_norm.loc[:,40:45][self.state_df['clusters_knn'] == i].mean().mean()
            state_code[i] = theta/delta * gamma

        if Nclusters == 3:
            sws_code = np.argsort(state_code)[0]
            wake_code = np.argsort(state_code)[1]
            rem_code = np.argsort(state_code)[2]

            conditions = [  (np.in1d(self.state_df['clusters_knn'], wake_code)),
                            (np.in1d(self.state_df['clusters_knn'], sws_code)),
                            (np.in1d(self.state_df['clusters_knn'], rem_code))]
            state_choices = ['Wake', 'SWS', 'REM']
            self.state_df['states'] = np.select(conditions, state_choices, default="ambiguous")
        elif Nclusters == 4:
            LMwake_code = np.argsort(state_code)[1]
            sws_code = np.argsort(state_code)[0]
            HMwake_code = np.argsort(state_code)[3]
            rem_code = np.argsort(state_code)[2]

            conditions = [ (np.in1d(self.state_df['clusters_knn'], HMwake_code)),
                           (np.in1d(self.state_df['clusters_knn'], LMwake_code)),
                           (np.in1d(self.state_df['clusters_knn'], sws_code)),
                           (np.in1d(self.state_df['clusters_knn'], rem_code))]

            state_choices = ['HMwake', 'LMwake', 'SWS', 'REM']
            self.state_df['states'] = np.select(conditions, state_choices, default="ambiguous")
        elif Nclusters > 4:
            real_clusters = np.argsort(np.unique(clone.labels_,return_counts=True))[1]
            LMwake_code = real_clusters[-3]
            sws_code = real_clusters[-2]
            HMwake_code = real_clusters[-1]
            rem_code = real_clusters[-4]

            conditions = [ (np.in1d(self.state_df['clusters_knn'], HMwake_code)),
                           (np.in1d(self.state_df['clusters_knn'], LMwake_code)),
                           (np.in1d(self.state_df['clusters_knn'], sws_code)),
                           (np.in1d(self.state_df['clusters_knn'], rem_code))]

            state_choices = ['HMwake', 'LMwake', 'SWS', 'REM']
            self.state_df['states'] = np.select(conditions, state_choices, default="ambiguous")
        else:
            print('Number of clusters not recognized. Automatic state assignment failed')



def density_calc(dataframe, boundary=(-100, 90)):
    # now calculate the bins for each frequency
    density_mat = []
    mean_density = []
    for i in range(len(dataframe.index)):
        # print(i)
        density, bins = np.histogram(dataframe.iloc[i, :], bins=5000, range=boundary, density=True)
        density_mat.append(density)
        mean_density.append(dataframe.iloc[i, :].mean())
    density_mat = np.array(density_mat)
    bins = (bins[1:] + bins[:-1]) / 2
    return density_mat, bins

def iterative_savitzky_golay(data,iterations=3):
    """
    This function calculates the Savitzky-Golay filtered EEG signal,
    which is used to correct artifacts caused by baseline shifts.
    """
    signal = data.EEG_data
    w = int(data.EEG_fs//2) #window size used for filtering
    if (w % 2)==0: # making sure window length is odd
        w+=1
    for i in range(iterations):
        print ('Removing drift in baseline: Iteration {}/3'.format(i+1))
        if i==0:
            signal_sg = scipy.signal.savgol_filter(signal,
                                    w, 2) # order of fitted polynomial
        else:
            signal_sg = scipy.signal.savgol_filter(signal_sg,
                                    w, 2) # order of fitted polynomial
    signal_corrected = signal - signal_sg
    return signal_sg , signal_corrected


def deg_overlap(n1,n2):
    try:
        n_feat = n1.shape[1]
    except IndexError:
        n_feat = 1
    n = np.concatenate((n1,n2),axis=0).reshape(-1,n_feat)
    # Construct Euclidean  Minimum  Spanning  Tree (EMST)
    tree = cKDTree(n)
    # find neighbors (includes identity)
    dist , idx = tree.query(n, k=2)
    # get the mapping to the closest non-self neighbor
    v1 = idx[:len(n1),1]
    # count how many belong to the first set
    m1 = len(v1[v1<=(len(n1)-1)])
    #calculate Degree of Overlap
    DO = min(1, (len(n)*(len(n1)-m1))/(len(n1)*len(n2)))
    return DO


def outliers_z_score(ys,threshold=5):
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y) / stdev_y for y in ys]
    outliers = np.where(np.abs(z_scores) > threshold)
    return outliers[0]



def deg_overlap(n1,n2):
    try:
        n_feat = n1.shape[1]
    except IndexError:
        n_feat = 1
    n = np.concatenate((n1,n2),axis=0).reshape(-1,n_feat)
    # Construct Euclidean  Minimum  Spanning  Tree (EMST)
    tree = cKDTree(n)
    # find neighbors (includes identity)
    dist , idx = tree.query(n, k=2, p=2)
    # get the mapping to the closest non-self neighbor
    v1 = idx[:len(n1),1]
    # count how many belong to the first set
    m1 = len(v1[v1<=(len(n1)-1)])
    #calculate Degree of Overlap
    try:
        DO = min(1, (len(n)*(len(n1)-m1))/(len(n1)*len(n2)))
    except ZeroDivisionError:
        DO = 0
    return DO


def CI95(data, confidence=0.95,axis=1):
    if axis==1:
        n = data.shape[axis]
        m = data.mean(axis=axis,skipna=True)
        std_err = scipy.stats.sem(data,axis=axis,nan_policy='omit')
        h = std_err * scipy.stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        print('data structure unknown')
    return m,h

##--------------------------
#return run lengths for each state
##--------------------------
def rle(inarray):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])
