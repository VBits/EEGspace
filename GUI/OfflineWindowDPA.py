"""
Online analysis
"""
import sys

from Pipeline import DPA
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QWidget, QComboBox, QLabel, QHBoxLayout, QGridLayout
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from GUI.PageWindow import PageWindow
from OfflineAnalysis.Utilities.GeneralUtils import get_random_idx
from OfflineAnalysis import Config as OfflineConfig

plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

class OfflineWindowDPA(PageWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("DPA Analysis")
        self.settings = QSettings("ClosedLoopEEG", "OfflineSettingsDPA")

        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(projection='3d')

        #todo edit this window
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
        # 4. Density peak clustering
        # Find density peaks in low dimensional space,
        mouse = self.state["mouse"]
        rand_idx = get_random_idx(mouse.Sxx_ext, size=OfflineConfig.random_epoch_size)
        dpa_z = OfflineConfig.dpa_z
        k_max = OfflineConfig.dpa_k_max
        est = DPA.DensityPeakAdvanced(Z=dpa_z, k_max=k_max)
        est.fit(mouse.LD_df.loc[rand_idx])

        # Optional: Repeat clustering with DPA by tweaking Z, number of standard deviations
        # Hint: some times it can also help to select a new rand_idx
        # est = clustering_DPA(m,rand_idx,dpa_z=0.6)

        # Optional: remap the spurious clusters into 4 labels
        label_dict = {0: [1, 4, 3, 6],
                      1: [0],
                      2: [2],
                      3: [5]}
        # change the dictionary format
        label_dict = {vi: k for k, v in label_dict.items() for vi in v}
        # merge the labels
        est.labels_ = np.vectorize(label_dict.get)(est.labels_)

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
