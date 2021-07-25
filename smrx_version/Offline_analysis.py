
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import sys
sys.path.append('C:/Users/bitsik0000/PycharmProjects/delta_analysis/SleepAnalysisPaper')
from smrx_version.functions import *
import matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
### -------------------
# -----------------------------------------------------------------
# Inputs
def run_offline_analysis():
    mh = Mouse("B6J", 1)

    #data directory
    EphysDir = 'C:/Users/matthew.grant/source/repos/ClosedLoopEEG/data/Ephys/'

    Folder = ''
    File = '210409_000.smrx'

    FilePath = EphysDir+Folder+File


    #Create directory to save figures
    figureFolder = mh.gen_folder(EphysDir,Folder)
    mh.figure_tail = ' - {} - {}.png'.format(mh.pos, mh.genotype)
    #Load data
    mh.read_smrx(FilePath)
    # standard functions for plotting
    plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
    # Set figure resolution
    dpi = 500
    # -----------------------------------------------------------------

    ###------------------------
    #OPTIONAL remove drifting baseline
    #This might be making 60Hz noise worse, so it needs to be called on demand
    # _ , mh.EEG_data = iterative_savitzky_golay(mh,iterations=3)


    #### -------------------
    # # evaluate quality of recordings
    matplotlib.use('Agg')
    plt.figure()
    plt.plot(mh.EEG_data)
    plt.title('{}'.format(mh.Ch_name))
    plt.ylabel(mh.Ch_units)
    plt.ylim(1000,-1000)
    plt.savefig(figureFolder+'{}_{}'.format(mh.Ch_name,File[:6]) + mh.figure_tail)
    matplotlib.use('Qt5Agg')

    ### -------------------
    ### Downsample, perform multitaper and normalize data
    target_fs=100
    if mh.EEG_fs > target_fs:
        print ('downsampling mouse {} EEG data, from {}Hz to {}Hz'.format(mh.pos,mh.EEG_fs,target_fs))
        mh.downsample_EGG(target_fs=target_fs)

    # resolution in seconds
    mh.multitaper(resolution=2)
    # specify amount of smoothing (depends also on resolution)
    mh.Sxx_norm_unsmoothed = mh.process_spectrum_detached(smooth_iter=0, window_size=41)

    mh.Sxx_norm = mh.process_spectrum_detached(smooth_iter=4, window_size=41)

    #Save normalized Dataframe to experimental folder
    #mh.Sxx_norm.to_pickle(EphysDir + Folder + 'Sxx_norm_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],mh.genotype,mh.pos))
    #Load previously saved Dataframe from experimental folder
    #mh.Sxx_norm = pd.read_pickle(EphysDir + Folder + 'Sxx_norm_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],mh.genotype,mh.pos))

    return mh
