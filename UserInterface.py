from PyQt5.QtWidgets import QWidget, QLabel,QHBoxLayout,QVBoxLayout,\
    QPushButton, QSlider, QStyle, QSizePolicy, QFileDialog, QGridLayout
from PyQt5.QtCore import  Qt, QUrl
from PyQt5.QtGui import QPalette,QIcon
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
import datetime
import os
sys.path.append('C:/Users/bitsik0000/PycharmProjects/delta_analysis/SleepAnalysisPaper')
from utils import *
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
### -------------------

class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.subplots()
        # self.ax.set_axis_off()

        # pC1 = np.random.random(100)
        # pC2 = np.random.random(100)
        initial_spot_x = np.random.random(1)
        initial_spot_y = np.random.random(1)
        # self.ax.imshow(Z)
        # self.ax.clear()
        # self.ax.scatter(pC1,pC2,c='black')
        self.ax.scatter(mh.pC_df['pC1'][rand_idx],mh.pC_df['pC2'][rand_idx], c='k', linewidths=0, alpha=0.4, s=4)
        self.spot = self.ax.scatter(initial_spot_x, initial_spot_y, c='r')
        # self.spot.remove()
        self.canvas.draw()
        # self.ax.scatter(np.random.random(10), np.random.random(10))
        self.video = Video()

        container = QWidget()
        layout = QGridLayout()
        layout.addWidget(self.canvas, 0, 1)
        layout.addWidget(self.video, 0, 0)
        container.setLayout(layout)
        self.setCentralWidget(container)

        # print (self.video.position_changed())
        # self.setCentralWidget(self.canvas)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.on_timeout)
        # timer interval in milliseconds
        timer.start(1000)


    def on_timeout(self):
        try:
            self.video.position
        except AttributeError:
            pass
        else:
            # find time for current position
            current_time = self.video.start_time + datetime.timedelta(milliseconds=self.video.position)
            print(current_time.strftime("%Y-%m-%d %H:%M:%S"))

            # search the dataframe for the closest value to display
            # Get a smaller dataframe with the data contained in the video

            # Search for the closest index
            position_idx = mh.pC_df.index.get_loc(current_time, method='nearest')

            # Get the PCA values for that index value
            spot_x = mh.pC_df.iloc[position_idx]['pC1']
            spot_y = mh.pC_df.iloc[position_idx]['pC2']
            # self.ax.imshow(Z)
            # self.ax.clear()
            #TODO get the position and update point
            # print(self.position)
            self.spot.remove()
            self.spot = self.ax.scatter(spot_x,spot_y,c='r')
            self.canvas.draw()

class Video(QWidget):
    def __init__(self):
        super().__init__()

        # self.setWindowTitle("PyQt5 Media Player")
        self.setGeometry(350, 100, 700, 500)
        self.setWindowIcon(QIcon('player.png'))

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.init_ui()

        self.show()

    def init_ui(self):

        # create media player object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # create videowidget object

        videowidget = QVideoWidget()

        # create open button
        openBtn = QPushButton('Open Video')
        openBtn.clicked.connect(self.open_file)

        # create button for playing
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)

        # create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)

        # create label
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # create hbox layout
        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0, 0, 0, 0)

        # set widgets to the hbox layout
        hboxLayout.addWidget(openBtn)
        hboxLayout.addWidget(self.playBtn)
        hboxLayout.addWidget(self.slider)

        # create vbox layout
        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(videowidget)
        vboxLayout.addLayout(hboxLayout)
        vboxLayout.addWidget(self.label)

        self.setLayout(vboxLayout)

        self.mediaPlayer.setVideoOutput(videowidget)

        # media player signals

        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    def open_file(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, "Open Video")
        print(self.filename)

        if self.filename != '':
            self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(self.filename)))
            self.playBtn.setEnabled(True)

        #get the start time of the video opened
        csv_file = self.filename[0:-4] + '.csv'
        csv_data = np.genfromtxt(csv_file, delimiter=',')
        start_time = csv_data[0, 2:3][0]
        # convert the timestamps to datetime objects
        moment = '{0:.0f}'.format(start_time)
        # Account for rounding errors
        if moment[10:] == '60000000':
            moment = moment[:10] + '59999999'
        self.start_time = datetime.datetime.strptime(moment, "%y%m%d%H%M%S%f")

    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()

        else:
            self.mediaPlayer.play()

    def mediastate_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause)

            )

        else:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)

            )

    def position_changed(self, position):
        self.slider.setValue(position)
        self.position = position

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("Error: " + self.mediaPlayer.errorString())

# -----------------------------------------------------------------
# Inputs

#TODO provide genotype and rig position
mh = Mouse("B6J",7)
#data directory
EphysDir = 'D:/Ongoing_analysis/'

#experiment directory and filename
Folder = '200424_B6J_Misting_NO/'
FileMat = '200526_B6J_misting_NO.mat'

# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}

# Set figure resolution
dpi = 500
# -----------------------------------------------------------------
#Create directory to save figures

figure_tail =  ' - {} - {}.png'.format(mh.pos,mh.genotype)
date = datetime.datetime.now().strftime("%y%m%d")
figureFolder = EphysDir + Folder + 'Mouse_{}_{}/'.format(mh.pos,date)
if not os.path.exists(figureFolder):
    print ('Directory created')
    os.makedirs(os.path.dirname(figureFolder), exist_ok=True)
else:
    print ('Directory exists')

# Load data
mh.add_data(EphysDir + Folder,FileMat)


###------------------------
# remove drifting baseline
x , y = iterative_savitzky_golay(mh,iterations=3)
mh.EEG_data = y

### -------------------
# Perform stft and downsample
nperseg = 3*mh.EEG_fs
mh.sleep_bandpower(nperseg=nperseg, fs = mh.EEG_fs, EMG=False,LP_filter=False)

#TODO
# #option 1
# mh.df = mh.df.rolling(11, center=True,win_type=None, min_periods=2).mean()
#option 2
def SG_filter(x):
    return scipy.signal.savgol_filter(x, 21, 2)

mh.df = mh.df.apply(SG_filter)
mh.LP_filter = True
###################################################
#---------------------------------------
# Make PCA, this function will calculate state space locations for these points
# based on a rolling mean window before normalization
mh.PCA()

rand_idx = np.random.choice(len(mh.pC), size=40000,replace=False)

fig = plt.figure()
plt.scatter(*mh.pC.T[:,rand_idx], c='k', linewidths=0, alpha=0.4, s=4)
plt.title('PCA')
plt.savefig(figureFolder+'PCA' + figure_tail, dpi=dpi)

mh.pC_df.to_pickle(figureFolder+'pC_df.pkl')
mh.pC_df = pd.read_pickle(figureFolder+'pC_df.pkl')
rand_idx = np.random.choice(len(mh.pC_df), size=40000,replace=False)




plot_app = QtWidgets.QApplication(sys.argv)
p = PlotWindow()
p.resize(1200, 700)
p.show()
plot_app.exec_()
plot_app.quit()