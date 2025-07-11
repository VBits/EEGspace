from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
import pandas as pd
import h5py
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


class Mouse:
    """
    # The mouse class object, holds data and information for each mouse
    """
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


    def add_data(self, Folder, FileMat):
        """
        Load data from .mat file format
        :param Folder: folder containing the EEG data
        :param FileMat: the .mat file
        :return:
        """
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

    def read_smrx(self, BaseDir, ExpDir, File):
        """
        Load EEG data from .smrx file using Spike2 Python SDK.
        Stores BaseDir, ExpDir, and File for use in other methods.
        Also creates self.FileHandle for the open file.
        """
        # Save path info
        self.BaseDir = BaseDir
        self.ExpDir = ExpDir
        self.File = File  # file name string

        # Other metadata
        self.figure_tail = ' - {} - {}.png'.format(self.pos, self.genotype)
        self.FilePath = BaseDir + ExpDir
        print(f'Loading Mouse {self.pos} from {self.FilePath}')

        # Open the .smrx file with Spike2 SDK
        self.FileHandle = sp.SonFile(self.FilePath + File, True)

        if self.FileHandle.GetOpenError() != 0:
            print('Error opening file:', sp.GetErrorString(self.FileHandle.GetOpenError()))
            quit()

        WaveChan = self.pos - 1
        self.Ch_units = self.FileHandle.GetChannelUnits(WaveChan)
        self.Ch_name = self.FileHandle.GetChannelTitle(WaveChan)

        # Duration
        dMaxSeconds = self.FileHandle.ChannelMaxTime(WaveChan) * self.FileHandle.GetTimeBase()
        dPeriod = self.FileHandle.ChannelDivide(WaveChan) * self.FileHandle.GetTimeBase()
        nPoints = floor(dMaxSeconds / dPeriod)

        self.EEG_data = np.array(self.FileHandle.ReadFloats(WaveChan, nPoints, 0))

        start = pd.DataFrame(np.reshape(self.FileHandle.GetTimeDate()[::-1][:-1], (1, 6)),
                             columns=['year', 'month', 'day', 'hour', 'minute', 'second'])
        self.start = pd.to_datetime(start)
        self.EEG_fs = 1 / dPeriod
        self.EEG_ideal_fs = self.FileHandle.GetIdealRate(WaveChan)

    def process_TTL_epochs(self, epoch_length_sec=2, fraction_threshold=0.1):
        """
        Detect TTL 'ON' epochs for square-wave TTL signal.
        Marks an epoch as ON if the fraction of samples >2.5 V exceeds threshold.
        """
        ttl = np.asarray(self.TTL_signal)
        epoch_samples = int(epoch_length_sec * self.TTL_fs)
        n_epochs = int(len(ttl) / epoch_samples)

        ttl_binary = ttl > 2.5

        ttl_on_epochs = []
        fraction_on_list = []

        for i in range(n_epochs):
            seg = ttl_binary[i * epoch_samples: (i + 1) * epoch_samples]
            fraction_on = np.mean(seg)
            fraction_on_list.append(fraction_on)
            ttl_on_epochs.append(fraction_on > fraction_threshold)

        df = pd.DataFrame({
            'epoch': np.arange(n_epochs),
            'TTL_on': ttl_on_epochs,
            'fraction_on': fraction_on_list
        })

        df['start_time_sec'] = df['epoch'] * epoch_length_sec

        # Add datetime for unambiguous merge
        df['datetime'] = df['start_time_sec'].apply(
            lambda x: self.start + pd.to_timedelta(x, unit='s')
        )

        self.TTL_epochs_df = df

        # Save
        fname = f'TTL_epochs_df_{self.ExpDir[:6]}_{self.File[:6]}_{self.genotype}_m{self.pos}.pkl'
        path = self.BaseDir + self.ExpDir + fname
        df.to_pickle(path)

        print(f'Saved TTL epochs DataFrame to: {path}')

    #Generate folder to store figures for mouse
    def gen_folder(self, BaseDir, ExpDir, all_mice=None):
        date = datetime.datetime.now().strftime("%y%m%d")
        if all_mice is None:
            self.figureFolder = BaseDir + ExpDir + 'Mouse_{}_{}/'.format(self.pos, date)
        else:
            self.figureFolder = BaseDir + ExpDir + 'All_Mice_{}/'.format(date)

        if not os.path.exists(self.figureFolder):
            print('Directory for m{} created'.format(self.pos))
            os.makedirs(os.path.dirname(self.figureFolder), exist_ok=True)
        else:
            print('Directory for m{} exists'.format(self.pos))


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
        self.multitaper_df = 10 * np.log(self.multitaper_df)

    #Smoothen the multitaper data with median filter
    def process_spectrum(self, window_size=21):
        """
        Smoothen the multitaper data with median filter. Non linear filtering preserves transitions better than
        savgol filtering
        :param window_size: 21 provides a good compromise between smoothing and retaining information
        :return:
        """
        self.Sxx_df = self.multitaper_df.rolling(window_size, center=True, win_type=None, min_periods=2).median()
        #normalize spectrum
        normalization = self.Sxx_df.quantile(q=0.01, axis=0)
        self.Sxx_norm = self.Sxx_df - normalization

    def knn_pred(self, clf, dataframe,state_averages_path):
        """
        Expand the DPC labels to the rest of the dataset using KNN
        :param clf:
        :param dataframe:
        :param state_averages_path:
        :return:
        """
        # predict states
        self.state_df = pd.DataFrame(index=dataframe.index)
        self.state_df['clusters_knn'] = clf.predict(self.LD_df)


        Nclusters = len(self.state_df['clusters_knn'].unique())

        #read previously calculated state averages (normalized data)
        state_averages = pd.read_pickle(state_averages_path)


        #compute knn state averages
        label_averages = pd.DataFrame()
        for label in np.unique(self.state_df['clusters_knn']):
            label_averages[label] = self.Sxx_norm.loc[self.state_df[self.state_df['clusters_knn'] == label].index].mean(axis=0)
        #determine which knn labels match each state
        if Nclusters == 4:
            state_averages = state_averages.drop(['Wake'], axis=1)
            state_dict = {}
            for state in ['SWS','REM','HTwake','LTwake']:
                state_correlations = label_averages.corrwith(state_averages[state])
                state_dict[int(state_correlations.idxmax())] = state
                print (state)
                print (state_correlations.idxmax())
                label_averages.drop(columns=state_correlations.idxmax(),inplace=True)

            self.state_df['states'] = self.state_df['clusters_knn']
            self.state_df.replace(to_replace={"states": dict(state_dict)},inplace=True)
        elif Nclusters == 3:
            state_averages = state_averages.drop(['HTwake', 'LTwake'], axis=1)
            state_dict = {}
            for state in ['SWS','REM','Wake']:
                state_correlations = label_averages.corrwith(state_averages[state])
                state_dict[int(state_correlations.idxmax())] = state
                print (state)
                print (state_correlations.idxmax())
                label_averages.drop(columns=state_correlations.idxmax(),inplace=True)

            self.state_df['states'] = self.state_df['clusters_knn']
            self.state_df.replace(to_replace={"states": dict(state_dict)},inplace=True)
        else:
            print('Number of clusters not recognized. Automatic state assignment failed')




