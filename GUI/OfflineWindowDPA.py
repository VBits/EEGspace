"""
Online analysis
"""
import sys

from Pipeline import DPA
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QWidget, QComboBox, QLabel, QHBoxLayout, QGridLayout, QSpacerItem, QSizePolicy, QFormLayout, \
    QLineEdit, QPushButton, QVBoxLayout
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PyQt5.QtGui import QColor, QPalette

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

        container = QWidget()

        self.LDA_origin_label = QLabel("Select LDA source:")

        self.right_panel = QWidget()

        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)

        for i in range(5):
            text_field_label = QLabel()
            text_field_label.setObjectName(f"text_field_{i}_label")
            text_field_label.setText(str(i))
            form_layout.setWidget(i + 1, QFormLayout.LabelRole, text_field_label)

            text_field_input = QLineEdit()
            text_field_input.setObjectName(f"text_field_{i}_input")
            form_layout.setWidget(i + 1, QFormLayout.FieldRole, text_field_input)

        # add buttons
        self.dpa_z_button = QPushButton("dpa-z")
        form_layout.setWidget(5, QFormLayout.FieldRole, self.dpa_z_button)

        self.rm_dpa_button = QPushButton("RM DPA")
        form_layout.setWidget(6, QFormLayout.FieldRole, self.rm_dpa_button)

        self.new_random_indexes_button = QPushButton("New random indexes")
        form_layout.setWidget(7, QFormLayout.FieldRole, self.new_random_indexes_button)

        self.merge_labels_button = QPushButton("Merge labels")
        form_layout.setWidget(8, QFormLayout.FieldRole, self.merge_labels_button)

        self.right_panel.setLayout(form_layout)

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

        # Create the top panel
        top_panel_layout = QGridLayout()
        top_panel = QSpacerItem(20, 40, QSizePolicy.Minimum)
        top_panel_layout.addItem(top_panel, 0, 0, 1, 2)

        # Create the center panel
        center_panel = QWidget()
        center_panel.setStyleSheet("background-color: rgb(255, 0, 0);")  # set background color to red
        center_layout = QHBoxLayout()
        center_layout.addWidget(self.canvas, 1)
        center_layout.addWidget(self.right_panel, 1)
        center_panel.setLayout(center_layout)
        top_panel_layout.addWidget(center_panel, 1, 0, 1, 2)

        # Create the bottom panel
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.backButton, 1)
        bottom_layout.addWidget(self.startButton, 1)
        top_panel_layout.addLayout(bottom_layout, 2, 0, 1, 2)

        container.setLayout(top_panel_layout)
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
