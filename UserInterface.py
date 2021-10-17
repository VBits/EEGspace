"""
Online analysis
"""
import sys
from PyQt5.QtWidgets import QWidget, QComboBox, QLabel,QHBoxLayout,QVBoxLayout,\
    QPushButton, QSlider, QStyle, QSizePolicy, QFileDialog, QGridLayout
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import Config

plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import Modelling
import numpy as np
sys.path.append('C:/Users/bitsik0000/PycharmProjects/delta_analysis/SleepAnalysisPaper')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class PlotWindow(QtWidgets.QMainWindow):
    def __init__(self, queue):
        super().__init__()

        self.setWindowTitle("Closed Loop EEG")

        self.figure1 = Figure(figsize=(5, 5))
        self.canvas1 = FigureCanvasQTAgg(self.figure1)
        self.ax1 = self.figure1.add_subplot(111)
        self.ax1.set_title("Raw data")
        self.ax1.set_xlim([0, 400])
        self.ax1.set_ylim([-300, 300])

        self.figure2 = Figure(figsize=(5, 5))
        self.canvas2 = FigureCanvasQTAgg(self.figure2)
        self.ax2 = self.figure2.add_subplot(111)
        self.ax2.set_title("Transformed data")
        self.ax2.set_xlim([0, 200])
        self.ax2.set_ylim([-5, 35])

        self.figure3 = Figure(figsize=(5, 5))
        self.canvas3 = FigureCanvasQTAgg(self.figure3)
        self.ax3 = self.figure3.add_subplot(111, projection='3d')

        self.queue = queue
        self.model = Modelling.get_model_for_mouse(1)#todo
        self.lda_encoded = self.model.lda.transform(self.model.training_data)

        n = 40000
        rand_idx = np.random.choice(len(self.lda_encoded), size=n, replace=False)
        subset = self.lda_encoded[rand_idx]
        subset_states = np.array([self.model.states[state] for state in self.model.training_data_states])[rand_idx]
        subset_sate_colors = [Config.state_colors[state] for state in subset_states]
        alpha = 0.3
        size = 0.8
        self.ax3.scatter(subset[:, 0], subset[:, 1], subset[:, 2], c=subset_sate_colors, s=size, alpha=alpha)
        self.ax3.set_xlabel('ld1')
        self.ax3.set_ylabel('ld2')
        self.ax3.set_zlabel('ld3')
        self.ax3.set_title("State space")

        self.spot = self.ax3.scatter(0, 0, 0, c='b', s=size, alpha=alpha)
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

        container = QWidget()

        self.mouse_select_label = QLabel("Mouse number:")

        self.mouse_select = QComboBox()
        self.mouse_select.addItems([str(num) for num in Config.mice_numbers])
        self.mouse_select.activated[str].connect(self.mouse_change)

        self.indicator_layout = QGridLayout()
        self.indicator_panel_stylesheet = "text-align: center; border: none; padding: 5px; font-size: 20px;";
        self.indicator_panels = []
        for mouse_num in Config.mice_numbers:
            indicator_panel = QLabel("Reading buffer for mouse " + str(mouse_num) + "...")
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
        layout.addWidget(self.canvas3, 0, 4, 6, 6)

        container.setLayout(layout)
        self.setCentralWidget(container)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.on_timeout)
        # timer interval in milliseconds
        timer.start(500)

        self.mouse_num = Config.mice_numbers[0]#initialize to the first mouse number

    def mouse_change(self, mouse_num):
        self.mouse_num = mouse_num
        self.ax1.clear()
        self.ax2.clear()
        #self.spot.remove()#broken

    def on_timeout(self):
        if not self.queue.empty():
            result = self.queue.get()

            class_name = result.standardized_class_name
            indicator_panel = self.indicator_panels[Config.mice_numbers.index(result.mouse_number)]
            indicator_panel.setText("Predicted class for mouse " + str(result.mouse_number) + " at timepoint" +
                                    str(result.time_point) + " :" + class_name)
            stylesheet = self.indicator_panel_stylesheet + "color: black; background-color: " \
                         + Config.state_colors[class_name]
            indicator_panel.setStyleSheet(stylesheet)

            if result.mouse_number is not self.mouse_num:
                return

            raw_data = result.raw_data
            self.ax1.clear()
            self.ax1.plot([x for x in range(1, len(raw_data) + 1)], [point for point in raw_data])
            self.ax1.set_title("Raw data")
            self.ax1.set_xlim([0, 400])
            self.ax1.set_ylim([-200, 200])
            self.canvas1.draw()

            data = result.transformed_data
            self.ax2.clear()
            self.ax2.plot([x for x in range(1, 202)], [point for point in data[0]])
            self.ax2.set_title("Transformed data")
            self.ax2.set_xlim([0, 200])
            self.ax2.set_ylim([-5, 35])
            self.canvas2.draw()

            scatter_point = result.lda_point[0]
            self.spot.remove()
            self.spot = self.ax3.scatter(scatter_point[0], scatter_point[1], scatter_point[2], c='r', s=60)
            self.canvas3.draw()


def create_user_interface(queue):
    plot_app = QtWidgets.QApplication(sys.argv)
    p = PlotWindow(queue)
    p.resize(1200, 700)
    p.show()
    plot_app.exec_()
    plot_app.quit()