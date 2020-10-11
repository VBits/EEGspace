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
import inspect
from tslearn.preprocessing import TimeSeriesResampler
import Config
import pickle

def inspect_function(f):
    code, line_no = inspect.getsourcelines(f)
    print(''.join(code))

def create_mouse_object(mouse_num, mouse_object_path):
    # data directory
    EphysDir = 'D:/Ongoing_analysis/' if Config.is_vassilis_workstation else 'C:/Source/ClosedLoopEEG/'
    # experiment directory and filename
    Folder = '181008_TRAP_females_4/baseline/'
    FileMat = '181008_000_baseline.mat'
    # provide genotype and rig position
    mh = Mouse("TRAP", mouse_num)

    # -----------------------------------------------------------------
    # Load data
    mh.add_data(EphysDir + Folder, FileMat)

    nperseg = 4 * mh.EEG_fs
    mh.sleep_bandpower(nperseg=nperseg, fs=mh.EEG_fs, EMG=False, LP_filter=True, iterations=1)

    ###################################################
    # ---------------------------------------
    # PCA, this function will calculate state space distribution of all points
    mh.PCA(normalizer=False, robust=True)

    f = open(mouse_object_path, 'wb')
    pickle.dump(mh, f)

# The mouse class object ()
class Mouse:
    def __init__(self, genotype, pos):
        self.genotype = genotype  # instance variable unique to each instance
        self.pos = pos
        self.colors ={'Wake': '#80a035',  # green
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
        pos_test = "G{}".format(self.pos)
        self.Mouse_Ch = [s for s in self.Ch_name if pos_test in s]
        self.EEG_data = self.f[str(self.Mouse_Ch[0])]["values"][0, :]
        self.EEG_data = scipy.signal.resample(self.EEG_data, int(len(self.EEG_data)/2.5))

        start = pd.DataFrame(self.f['file']['start'][0].reshape(6,1).T, columns = ['year',
                              'month','day','hour','minute','second'])
        self.start = pd.to_datetime(start)
        self.interval = self.f["{}".format(self.Mouse_Ch[0])]['interval'][0][0]
        self.EEG_fs = int((1 / self.f["{}".format(self.Mouse_Ch[0])]['interval'][0][0])/2.5) #downsampled by a factor of 2.5
        if len(self.Mouse_Ch) == 2:
            self.EMG_data = self.f["{}".format(self.Mouse_Ch[1])]["values"][0, :]
            self.EMG_fs = 1 / self.f["{}".format(self.Mouse_Ch[1])]['interval'][0][0]

#TODO correct rounding errors that prevent EEG and EMG from EMG from matching
    def sleep_bandpower(self, x=None, fs=None, nperseg=None, EMG = False, LP_filter = False,iterations=2, mode="interp"):
        if x is None:
            x = self.EEG_data
        if fs is None:
            fs = self.EEG_fs
        if nperseg is None:
            nperseg = 4*self.EEG_fs
        #Each window is ~3sec (with 50% overlap), so approx 1.5sec
        self.f, self.t, Sxx = scipy.signal.spectrogram(x, fs=fs, noverlap=nperseg // 2, nperseg=int(nperseg), window='hamming',mode='psd')
        #measure in dB
        self.Sxx = 10*np.log(Sxx)
        ind_Dmin = np.argmax(self.f >= 1)
        ind_Dmax = np.argmax(self.f >= 4)
        ind_Tmin = np.argmax(self.f >= 7)  # 7
        ind_Tmax = np.argmax(self.f >= 10)  # 10
        ind_Smin = np.argmax(self.f >= 12)
        ind_Smax = np.argmax(self.f >= 18)
        ind_Gmin = np.argmax(self.f >= 32)
        ind_Gmax = np.argmax(self.f >= 45)
        if fs > 200:
            ind_fGmin = np.argmax(self.f >= 75)
            ind_fGmax = np.argmax(self.f >= 120)
        ms_freq = np.round(np.mean(np.ediff1d(self.t)) * 1000, 2)
        df_tim = pd.date_range(start=self.start[0], freq='{}ms'.format(ms_freq), periods=len(self.t))
        self.df = pd.DataFrame(index=df_tim)
        self.df['D_band'] = bn.nanmean(Sxx[ind_Dmin:ind_Dmax, :], axis=0)
        self.df['T_band'] = bn.nanmean(Sxx[ind_Tmin:ind_Tmax, :], axis=0)
        self.df['S_band'] = bn.nanmean(Sxx[ind_Smin:ind_Smax, :], axis=0)
        self.df['G_band'] = bn.nanmean(Sxx[ind_Gmin:ind_Gmax, :], axis=0)
        if fs > 200:
            self.df['fG_band'] = bn.nanmean(Sxx[ind_fGmin:ind_fGmax, :], axis=0)
        self.df['T_D_band'] = np.divide(self.df['T_band'], self.df['D_band'])
        self.df['T_G_band'] = np.divide(self.df['T_band'], self.df['G_band'])
        self.LP_filter = False
        if EMG:
            EMG_rms = np.sqrt(self.EMG_data ** 2)
            resampled_EMG = TimeSeriesResampler(sz=self.df.shape[0]).fit_transform(EMG_rms)
            self.df['EMG_rms'] = resampled_EMG.flatten()
        if scipy.signal.spectrogram:
            # deprecated, filter with rolling mean
            # self.df = self.df.rolling(window_size, center=True,win_type=None, min_periods=2).mean()
            def SG_filter(x):
                return scipy.signal.savgol_filter(x, 21, 2, mode=mode)
            print('Smoothing filter set at {} iterations'.format(iterations))
            for i in range(iterations):
                self.df = self.df.apply(SG_filter)
            self.LP_filter = True

    def PCA(self, window_size = 11,normalizer=False,robust =False, scaler=None, saved_pca=None):
        if scaler is not None:
            self.x = scaler.transform(self.df)
        elif self.LP_filter:
            #check if there is a scaler saved, if yes then use this, if not then use the input variables
            if normalizer:
                self.scaler = Normalizer()
                self.x = self.scaler.fit_transform(self.df)
                print('Using Normalizer')
            elif robust:
                self.scaler = RobustScaler(quantile_range=(1, 99))
                self.x = self.scaler.fit_transform(self.df)
                print('Using Robust Scaler')
            else:
                self.scaler = StandardScaler()
                self.x = self.scaler.fit_transform(self.df)
                print('Using Standard Scaler')
        else:
            self.scaler = StandardScaler()
            self.x = StandardScaler().fit_transform(self.df.rolling(window_size, center=True,
                                                     win_type=None, min_periods=2).mean())
            print('Using Standard Scaler and rolling mean')
        print('LP filter was set to {}'.format(self.LP_filter))
        if saved_pca is not None:
            self.pC = saved_pca.transform(self.x)
        else:
            self.pca = PCA(n_components=2)
            self.pC = self.pca.fit_transform(self.x)
        self.pC_df = pd.DataFrame(data=self.pC,columns=['pC1','pC2'],index=self.df.index)

    def knn_pred(self, clf):
        # predict in 2D
        self.state_df = pd.DataFrame(index=self.df.index)
        self.state_df['clusters_knn'] = clf.predict(self.pC)

        Nclusters = len(self.state_df['clusters_knn'].unique())

        # Count state instances after finding which code has higher average T_D.
        # Descending order(REM, Wake, SWS)
        state_code = np.zeros(Nclusters)
        for i in range(Nclusters):
            state_code[i] = np.mean(self.df['T_D_band'][self.state_df['clusters_knn'] == i])

        if Nclusters == 3:
            sws_code = np.argsort(state_code)[0]
            wake_code = np.argsort(state_code)[1]
            rem_code = np.argsort(state_code)[2]

            conditions = [  (np.in1d(self.state_df['clusters_knn'], wake_code)),
                            (np.in1d(self.state_df['clusters_knn'], sws_code)),
                            (np.in1d(self.state_df['clusters_knn'], rem_code))]
        elif Nclusters == 4:
            sws_code = np.argsort(state_code)[0]
            LAwake_code = np.argsort(state_code)[1]
            HAwake_code = np.argsort(state_code)[2]
            rem_code = np.argsort(state_code)[3]

            conditions = [ (np.in1d(self.state_df['clusters_knn'], [LAwake_code, HAwake_code])),
                           (np.in1d(self.state_df['clusters_knn'], sws_code)),
                           (np.in1d(self.state_df['clusters_knn'], rem_code))]
        else:
            print('Number of clusters not recognized. Run DPC again')

        state_choices = ['Wake', 'SWS', 'REM']

        self.state_df['3_states'] = np.select(conditions, state_choices, default="ambiguous")


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