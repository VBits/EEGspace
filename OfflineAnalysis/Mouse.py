from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
import pandas as pd
import h5py
import scipy
import scipy.signal
import bottleneck as bn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import Normalizer
from scipy.spatial import cKDTree
from scipy.signal import decimate, butter, dlti
import inspect
from tslearn.preprocessing import TimeSeriesResampler
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
                      'HTwake': '#80a035',
                     'LMwake':'#617D21',
                      'LTwake': '#617D21',
                  'Sleep': '#353377',  # blue
                  'SWS': '#353377',  # blue
                  'REM': '#aa6339',  # orange
                  'ambiguous': '#ff0000',  # red
                  }
        self.figure_tail = ' - {} - {}.png'.format(self.pos, self.genotype)

    def __repr__(self):
        return "Mouse in position {}, genotype {}".format(self.pos, self.genotype)

    #Load file if in .mat format
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

    #Load file if in .smrx format
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

    #Generate folder to store figures for mouse
    def gen_folder(self,EphysDir,Folder,all_mice=None):
        date = datetime.datetime.now().strftime("%y%m%d")
        if all_mice is None:
            figureFolder = EphysDir + Folder + 'Mouse_{}_{}/'.format(self.pos, date)
        else:
            figureFolder = EphysDir + Folder + 'All_Mice_{}/'.format(date)
        if not os.path.exists(figureFolder):
            print('Directory for m{} created'.format(self.pos))
            os.makedirs(os.path.dirname(figureFolder), exist_ok=True)
        else:
            print('Directory for m{} exists'.format(self.pos))
        return figureFolder

    #Downsample EEG and
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

    #Multitaper method for power spectrum estimation
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

    #Smoothen the multitaper data with Savgol
    def process_spectrum(self,smooth_iter=1, window_size=21,polynomial=4):
        ## Normalize the data and plot density spectrogram
        def SG_filter(x):
            return scipy.signal.savgol_filter(x, window_size, polynomial)

        # Log scale
        Sxx_df = 10 * np.log(self.multitaper_df.T)

        # horizontal axis (time)
        for i in range(smooth_iter):
            Sxx_df = Sxx_df.apply(SG_filter, axis=1, result_type='expand')

        self.Sxx_df = pd.DataFrame(data=Sxx_df.T.values, columns=self.multitaper_df.columns,
                                     index=self.multitaper_df.index)

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

    #Expand the sample labels to the rest of the data using
    def knn_pred(self, clf, Sxx_extended,state_averages_path):
        # predict states
        self.state_df = pd.DataFrame(index=Sxx_extended.index)
        self.state_df['clusters_knn'] = clf.predict(self.LD_df)


        Nclusters = len(self.state_df['clusters_knn'].unique())

        #read previously calculated state averages (normalized data)
        state_averages = pd.read_pickle(state_averages_path)
        #normalize spectrum
        normalization = self.Sxx_df.quantile(q=0.01, axis=0)
        self.Sxx_df_norm = self.Sxx_df - normalization

        #compute knn state averages
        label_averages = pd.DataFrame()
        for label in np.unique(self.state_df['clusters_knn']):
            label_averages[label] = self.Sxx_df_norm.loc[self.state_df[self.state_df['clusters_knn'] == label].index].mean(axis=0)
        #determine which knn labels match each state
        if Nclusters == 4:
            state_dict = {}
            for state in state_averages:
                state_correlations = label_averages.corrwith(state_averages[state])
                state_dict[state_correlations.argmax()] = state

            self.state_df['states'] = self.state_df['clusters_knn']
            self.state_df.replace({"states": state_dict},inplace=True)
        else:
            print('Number of clusters not recognized. Automatic state assignment failed')




