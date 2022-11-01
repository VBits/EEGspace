"""
Online analysis
"""
import sys

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QWidget, QComboBox, QLabel,QHBoxLayout,QVBoxLayout, \
    QGridLayout
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from GUI.PageWindow import PageWindow
from OnlineAnalysis import Config

plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

class OfflineWindowKnn(PageWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Knn Settings")
        self.settings = QSettings("ClosedLoopEEG", "OfflineSettingsKnn")

        self.figure = Figure(figsize=(5, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')

        mouse = self.state["mouse"]

        n = 40000
        rand_idx = np.random.choice(len(self.lda_encoded), size=n, replace=False)
        subset = self.lda_encoded[rand_idx]
        subset_states = np.array([self.model.states[state] for state in self.model.training_data_states])[rand_idx]
        subset_sate_colors = [Config.state_colors[state] for state in subset_states]
        alpha = 0.3
        size = 0.8
        self.ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c=subset_sate_colors, s=size, alpha=alpha)
        self.ax.set_xlabel('ld1')
        self.ax.set_ylabel('ld2')
        self.ax.set_zlabel('ld3')
        self.ax.set_title("State space")

        self.spot = self.ax.scatter(0, 0, 0, c='b', s=size, alpha=alpha)
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas.draw()

        container = QWidget()

        self.mouse_select_label = QLabel("Mouse number:")

        self.mouse_select = QComboBox()
        self.mouse_select.addItems([str(num) for num in Config.mouse_ids])
        self.mouse_select.activated[str].connect(self.mouse_change)

        self.indicator_layout = QGridLayout()
        self.indicator_panel_stylesheet = "text-align: center; border: none; padding: 5px; font-size: 20px;";
        self.indicator_panels = []
        for mouse_id in Config.mouse_ids:
            indicator_panel = QLabel("Reading buffer for mouse " + str(mouse_id) + "...")
            indicator_panel.setStyleSheet(self.indicator_panel_stylesheet)
            indicator_panel.setAlignment(QtCore.Qt.AlignCenter)
            self.indicator_panels.append(indicator_panel)
            self.indicator_layout.addWidget(indicator_panel)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(self.mouse_select_label, 1)
        selector_layout.addWidget(self.mouse_select, 5)

        left_panel = QVBoxLayout()
        left_panel.addLayout(selector_layout)
        left_panel.addLayout(self.indicator_layout)

        layout = QGridLayout()
        layout.addLayout(left_panel, 0, 0, 2, 4)
        layout.addWidget(self.canvas1, 2, 0, 2, 4)
        layout.addWidget(self.canvas2, 4, 0, 2, 4)
        layout.addWidget(self.canvas, 0, 4, 6, 6)

        container.setLayout(layout)
        self.setCentralWidget(container)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.on_timeout)
        # timer interval in milliseconds
        timer.start(500)

    def plot_data(self):
        pass
        #dropdown options

    def on_timeout(self):
        pass
