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
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
import datetime
import os
sys.path.append('C:/Users/bitsik0000/PycharmProjects/delta_analysis/SleepAnalysisPaper')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
### -------------------

class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, queue, lda_for_plot):
        super().__init__()
        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.queue = queue
        # self.ax.set_axis_off()

        # initial_spot_x = np.random.random(1)
        # initial_spot_y = np.random.random(1)
        # self.ax.imshow(Z)
        # self.ax.clear()
        self.ax2.scatter(lda_for_plot, c='black')

        #self.ax.scatter(mh.pC_df['pC1'][rand_idx],mh.pC_df['pC2'][rand_idx],)
        self.spot = self.ax.scatter([0, 0, 0], c='r')
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
            scatter_point = result.lda_point
            self.ax1.clear()
            self.ax1.plot([x for x in range(1, 202)], [point for point in data[0]])
            self.canvas.draw()
            self.spot.remove()
            self.spot = self.ax.scatter(scatter_point, c='r')


def create_user_interface(queue, lda_for_plot):
    plot_app = QtWidgets.QApplication(sys.argv)
    p = PlotWindow(queue, lda_for_plot)
    p.resize(1200, 700)
    p.show()
    plot_app.exec_()
    plot_app.quit()