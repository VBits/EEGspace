import sys
from PyQt5.QtWidgets import QWidget, QLabel,QHBoxLayout,QVBoxLayout,\
    QPushButton, QSlider, QStyle, QSizePolicy, QFileDialog, QGridLayout
from PyQt5.QtCore import  Qt, QUrl
from PyQt5.QtGui import QPalette,QIcon
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import Modelling
import numpy as np
import datetime
import os
sys.path.append('C:/Users/bitsik0000/PycharmProjects/delta_analysis/SleepAnalysisPaper')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
### -------------------

class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, queue):
        super().__init__()
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212, projection='3d')
        self.queue = queue
        self.model = Modelling.get_model_for_mouse(1)#todo
        self.lda_encoded = self.model.lda.transform(self.model.training_data)
        # self.ax.set_axis_off()

        # initial_spot_x = np.random.random(1)
        # initial_spot_y = np.random.random(1)
        # self.ax.imshow(Z)
        # self.ax.clear()
        # self.ax2.scatter(lda_for_plot, c='black')

        n = 40000
        rand_idx = np.random.choice(len(self.lda_encoded), size=n, replace=False)
        subset = self.lda_encoded[rand_idx]
        self.ax2.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c='b', s=1)
        self.ax2.set_xlabel('ld1')
        self.ax2.set_ylabel('ld2')
        self.ax2.set_zlabel('ld3')

        #self.ax.scatter(mh.pC_df['pC1'][rand_idx],mh.pC_df['pC2'][rand_idx],)
        self.spot = self.ax2.scatter(0, 0, 0, c='r')
        self.canvas.draw()
        # self.ax.scatter(np.random.random(10), np.random.random(10))
        #self.video = Video()

        container = QWidget()
        layout = QGridLayout()
        layout.addWidget(self.canvas, 0, 1)
        # layout.addWidget(self.video, 0, 0)
        container.setLayout(layout)
        self.setCentralWidget(container)

        # print (self.video.position_changed())
        # self.setCentralWidget(self.canvas)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.on_timeout)
        # timer interval in milliseconds
        timer.start(500)


    def on_timeout(self):
        if not self.queue.empty():
            result = self.queue.get()
            data = result.transformed_data
            self.ax1.clear()
            self.ax1.plot([x for x in range(1, 202)], [point for point in data[0]])
            scatter_point = result.lda_point[0]
            self.spot.remove()
            self.spot = self.ax2.scatter(scatter_point[0], scatter_point[1], scatter_point[2], c='r')
            self.canvas.draw()


def create_user_interface(queue):
    plot_app = QtWidgets.QApplication(sys.argv)
    p = PlotWindow(queue)
    p.resize(1200, 700)
    p.show()
    plot_app.exec_()
    plot_app.quit()