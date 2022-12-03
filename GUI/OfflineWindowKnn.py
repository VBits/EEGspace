"""
Online analysis
"""
import sys

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QWidget, QComboBox, QLabel, QHBoxLayout, QVBoxLayout, \
    QGridLayout
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from GUI.PageWindow import PageWindow
from OfflineAnalysis.Utilities.GeneralUtils import get_random_idx
from OfflineAnalysis.Utilities.LoadData import get_LDA
from OfflineAnalysis import Config as OfflineConfig

plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

class OfflineWindowKnn(PageWindow):
    def __init__(self):
        #add in a panel for the dropdown and one for the plotting window
        #Add the dropdown options, the go button
        #
        super().__init__()

        self.setWindowTitle("KNN Settings")
        self.settings = QSettings("ClosedLoopEEG", "OfflineSettingsKNN")

        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(projection='3d')

        container = QWidget()

        self.LDA_origin_label = QLabel("Select KNN source:")

        self.lda_origin_combobox = QComboBox()
        self.lda_origin_combobox.addItems([
            "Load previously trained KNN",
            "Train KNN on DPA labels"
        ])

        self.plot_data_button = QtWidgets.QPushButton("Load and plot data", self)
        self.plot_data_button.setGeometry(QtCore.QRect(5, 5, 200, 20))

        self.save_eeg_button = QtWidgets.QPushButton("Save EEG", self)
        self.save_eeg_button.setGeometry(QtCore.QRect(5, 5, 200, 20))
        self.save_eeg_button.clicked.connect(self.save_eeg)

        self.propagate_labels_button = QtWidgets.QPushButton("Propagate labels", self)
        self.propagate_labels_button.setGeometry(QtCore.QRect(5, 5, 200, 20))
        self.propagate_labels_button.clicked.connect(self.propagate_labels)

        self.indicator_layout = QGridLayout()

        settings_layout = QHBoxLayout()
        settings_layout.addWidget(self.LDA_origin_label, 1)
        settings_layout.addWidget(self.lda_origin_combobox, 5)
        settings_layout.addWidget(self.plot_data_button, 12)

        layout = QGridLayout()
        layout.addLayout(settings_layout, 0, 0, 2, 4)
        layout.addWidget(self.canvas, 2, 0, 2, 4)

        container.setLayout(layout)
        self.setCentralWidget(container)

    def plot_data(self, mouse):
        use_lda = self.lda_origin_combobox.currentIndex() + 1
        rand_idx = get_random_idx(mouse.Sxx_ext, size=OfflineConfig.random_epoch_size)
        _ = get_LDA(mouse, rand_idx, use_lda)
        lda_encoded = mouse.LD_df
        LD_rand = lda_encoded.loc[rand_idx]
        alpha = 0.3
        size = 0.8
        ax = self.ax
        if LD_rand.shape[1] == 3:
            ax.clear()
            ax.scatter(LD_rand['LD1'], LD_rand['LD2'], LD_rand['LD3'],
                       c='k', alpha=alpha, s=size, linewidths=0)
            ax.set_xlabel('LD1')
            ax.set_ylabel('LD2')
            ax.set_zlabel('LD3')
        else:
            plt.figure()
            plt.scatter(LD_rand['LD1'], LD_rand['LD2'],
                        c='k', alpha=alpha, s=size, linewidths=0)
            plt.xlabel('LD1')
            plt.ylabel('LD2')

        ax.set_title("LDA")
        self.canvas.draw()

    def propagate_labels(self):
        pass

    def save_eeg(self):
        pass

    def mount(self):
        print(self.state)
        print(self.state.keys())
        if "mouse" not in self.state:
            return
        mouse = self.state["mouse"]
        self.plot_data_button.clicked.connect(lambda: self.plot_data(mouse))
        self.plot_data(mouse)
