"""
Online analysis
"""
import sys

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QWidget, QComboBox, QLabel, QHBoxLayout, QVBoxLayout, \
    QGridLayout, QPushButton
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
import numpy as np
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

class OfflineWindowLDA(PageWindow):
    def __init__(self):
        #add in a panel for the dropdown and one for the plotting window
        #Add the dropdown options, the go button
        #
        super().__init__()

        self.setWindowTitle("LDA Settings")
        self.settings = QSettings("ClosedLoopEEG", "OfflineSettingsLDA")

        #main widget
        #QV widget for the bottom bar and the top widget
        #inside the top widget there should be QH widget for the plot panel on the left and the form panel on the right
        #form panel needs a value for the max and the dpa-z
        #checkboxes for the new rand and merge labels
        #list of 4 labels
        #and update button

        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(projection='3d')

        container = QWidget()

        self.LDA_origin_label = QLabel("Select LDA source:")

        self.lda_origin_combobox = QComboBox()
        load_average_option = "Load average LDA trained on multiple animals"
        load_previously_trained_option = "Load previously trained LDA for this animal"
        get_provisional_labels_option = "Get provisional labels from ANN and train a new LDA"
        self.lda_origin_combobox.addItems([
            load_average_option,
            load_previously_trained_option,
            get_provisional_labels_option
        ])

        self.plot_data_button = QtWidgets.QPushButton("Load and plot data", self)
        self.plot_data_button.setGeometry(QtCore.QRect(5, 5, 200, 20))
        self.plot_data_button.clicked.connect(self.plot_data)

        self.save_eeg_button = QtWidgets.QPushButton("Save EEG", self)
        self.save_eeg_button.setGeometry(QtCore.QRect(5, 5, 200, 20))
        self.save_eeg_button.clicked.connect(self.save_eeg)

        self.startButton = QtWidgets.QPushButton("Start")
        self.startButton.setGeometry(QtCore.QRect(450, 320, 151, 28))
        self.startButton.clicked.connect(self.go_to_next)
        self.backButton = QtWidgets.QPushButton("Back")
        self.backButton.setGeometry(QtCore.QRect(20, 640, 93, 28))
        self.backButton.clicked.connect(self.go_back)

        self.indicator_layout = QGridLayout()

        settings_layout = QHBoxLayout()
        settings_layout.addWidget(self.LDA_origin_label, 1)
        settings_layout.addWidget(self.lda_origin_combobox, 5)
        settings_layout.addWidget(self.plot_data_button, 12)

        navigation_layout = QHBoxLayout()
        navigation_layout.addWidget(self.backButton, 1)
        navigation_layout.addWidget(self.startButton, 1)

        layout = QGridLayout()
        layout.addLayout(settings_layout, 0, 0, 2, 4)
        layout.addWidget(self.canvas, 2, 0, 2, 4)
        layout.addLayout(navigation_layout, 4, 0, 2, 4)

        container.setLayout(layout)
        self.setCentralWidget(container)

    def plot_data(self):
        mouse = self.state["mouse"]
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

    def go_to_next(self):
        self.goto("dpa_window")

    def go_back(self):
        self.goto("offline_settings")

    def save_eeg(self):
        pass

    def mount(self):
        if "mouse" not in self.state:
            return
        self.plot_data()
