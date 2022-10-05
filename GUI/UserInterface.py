"""
Online analysis
"""
import sys
from PyQt5 import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from OnlineAnalysis import Config
from OnlineAnalysis.LoadModels import MouseModel
from types import SimpleNamespace

plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
sys.path.append('C:/Users/bitsik0000/PycharmProjects/delta_analysis/SleepAnalysisPaper')
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


#window for the master page window
class Window(QtWidgets.QMainWindow):
    def __init__(self, queue, config_queue, parent=None):
        super().__init__(parent)

        self.stacked_widget = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.m_pages = {}

        self.register(StartWindow(), "start")
        self.register(OnlineSettingsWindow(config_queue), "online_settings")
        self.register(PlotWindow(queue), "plot")
        self.register(OfflineSettingsWindow(), "offline_settings")
        self.register(ModelCreationWindow(), "offline_settings")

        self.goto("start")

    def register(self, widget, name):
        self.m_pages[name] = widget
        self.stacked_widget.addWidget(widget)
        if isinstance(widget, PageWindow):
            widget.gotoSignal.connect(self.goto)

    @QtCore.pyqtSlot(str)
    def goto(self, name):
        if name in self.m_pages:
            widget = self.m_pages[name]
            self.stacked_widget.setCurrentWidget(widget)
            self.setWindowTitle(widget.windowTitle())


class PageWindow(QtWidgets.QMainWindow):
    gotoSignal = QtCore.pyqtSignal(str)

    def goto(self, name):
        self.gotoSignal.emit(name)


#window for start page with two options, create the model or run a closed loop experiment with existing model => StartWindow
class StartWindow(PageWindow):
    def __init__(self):
        super().__init__()

        self.offlineButton = QtWidgets.QPushButton("Offline Analysis", self)
        self.offlineButton.setGeometry(QtCore.QRect(5, 5, 100, 20))
        self.offlineButton.clicked.connect(self.goToOfflineSettings)
        self.onlineButton = QtWidgets.QPushButton("Closed Loop Experiment", self)
        self.onlineButton.setGeometry(QtCore.QRect(110, 5, 100, 20))
        self.onlineButton.clicked.connect(self.goToOnlineSettings)

    def goToOfflineSettings(self):
        self.goto("offline_settings")

    def goToOnlineSettings(self):
        self.goto("online_settings")


class FileEditWidget(QtWidgets.QWidget):
    """
    A textfield with a browse button.

    """
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.file_line_edit = QtWidgets.QLineEdit()
        layout.addWidget(self.file_line_edit, stretch=1)
        browse_button = QtWidgets.QPushButton("...")
        layout.addWidget(browse_button)
        self.setLayout(layout)

        browse_button.clicked.connect(self.browse)

    def browse(self, msg = None, start_path = None):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self,
            msg or "Find files", start_path or QtCore.QDir.currentPath())
        if directory:
            self.file_line_edit.setText(directory)

    def setText(self, text):
        return self.file_line_edit.setText(text)

    def text(self):
        return self.file_line_edit.text()

#window for online mode settings, use QSettings and with a back button to the first screen => OnlineSettingsWindow
class OnlineSettingsWindow(PageWindow):
    def __init__(self, config_queue):
        super().__init__()
        self.settings = QSettings("ClosedLoopEEG", "OnlineSettings")
        self.resize(1117, 892)
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName(u"centralwidget")
        self.formLayoutWidget = QWidget(self.centralwidget)
        self.formLayoutWidget.setObjectName(u"formLayoutWidget")
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 30, 1071, 681))
        self.formLayout = QFormLayout(self.formLayoutWidget)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.eegFrequencyLabel = QLabel(self.formLayoutWidget)
        self.eegFrequencyLabel.setObjectName(u"eegFrequencyLabel")

        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.eegFrequencyLabel)

        self.eegFrequencyInput = QLineEdit(self.formLayoutWidget)
        self.eegFrequencyInput.setObjectName(u"eegFrequencyInput")

        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.eegFrequencyInput)

        self.downsampleFrequencyLabel = QLabel(self.formLayoutWidget)
        self.downsampleFrequencyLabel.setObjectName(u"downsampleFrequencyLabel")

        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.downsampleFrequencyLabel)

        self.downsampleFrequencyInput = QLineEdit(self.formLayoutWidget)
        self.downsampleFrequencyInput.setObjectName(u"downsampleFrequencyLabel_2")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.downsampleFrequencyInput)

        self.secondsLabel = QLabel(self.formLayoutWidget)
        self.secondsLabel.setObjectName(u"secondsLabel")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.secondsLabel)

        self.secondsPerEpochInput = QLineEdit(self.formLayoutWidget)
        self.secondsPerEpochInput.setObjectName(u"secondsPerEpochInput")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.secondsPerEpochInput)

        self.mouseIdLabel = QLabel(self.formLayoutWidget)
        self.mouseIdLabel.setObjectName(u"mouseIdLabel")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.mouseIdLabel)

        self.mouseIdInput = QLineEdit(self.formLayoutWidget)
        self.mouseIdInput.setObjectName(u"mouseIdInput")

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.mouseIdInput)

        self.bufferLengthLabel = QLabel(self.formLayoutWidget)
        self.bufferLengthLabel.setObjectName(u"bufferLengthLabel")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.bufferLengthLabel)

        self.bufferLengthInput = QLineEdit(self.formLayoutWidget)
        self.bufferLengthInput.setObjectName(u"bufferLengthInput")

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.bufferLengthInput)

        self.testDataLabel = QLabel(self.formLayoutWidget)
        self.testDataLabel.setObjectName(u"testDataLabel")

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.testDataLabel)

        self.rawTestDataFileInput = FileEditWidget(self.formLayoutWidget)
        self.rawTestDataFileInput.setObjectName(u"rawTestDataFileInput")

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.rawTestDataFileInput)

        self.runNameLabel = QLabel(self.formLayoutWidget)
        self.runNameLabel.setObjectName(u"runNameLabel")

        self.formLayout.setWidget(6, QFormLayout.LabelRole, self.runNameLabel)

        self.runNameInput = FileEditWidget(self.formLayoutWidget)
        self.runNameInput.setObjectName(u"runNameInput")

        self.formLayout.setWidget(6, QFormLayout.FieldRole, self.runNameInput)

        self.trainingDataFileLabel = QLabel(self.formLayoutWidget)
        self.trainingDataFileLabel.setObjectName(u"trainingDataFileLabel")

        self.formLayout.setWidget(7, QFormLayout.LabelRole, self.trainingDataFileLabel)

        self.trainingDataFileInput = FileEditWidget(self.formLayoutWidget)
        self.trainingDataFileInput.setObjectName(u"trainingDataFileInput")

        self.formLayout.setWidget(7, QFormLayout.FieldRole, self.trainingDataFileInput)

        self.trainingStatesFile = QLabel(self.formLayoutWidget)
        self.trainingStatesFile.setObjectName(u"trainingStatesFile")

        self.formLayout.setWidget(8, QFormLayout.LabelRole, self.trainingStatesFile)

        self.trainingStatesFileInput = FileEditWidget(self.formLayoutWidget)
        self.trainingStatesFileInput.setObjectName(u"trainingStatesFileInput")

        self.formLayout.setWidget(8, QFormLayout.FieldRole, self.trainingStatesFileInput)

        self.ldaFilePathLabel = QLabel(self.formLayoutWidget)
        self.ldaFilePathLabel.setObjectName(u"ldaFilePathLabel")

        self.formLayout.setWidget(9, QFormLayout.LabelRole, self.ldaFilePathLabel)

        self.ldaFilePathInput = FileEditWidget(self.formLayoutWidget)
        self.ldaFilePathInput.setObjectName(u"ldaFilePathInput")

        self.formLayout.setWidget(9, QFormLayout.FieldRole, self.ldaFilePathInput)

        self.knnFilePathLabel = QLabel(self.formLayoutWidget)
        self.knnFilePathLabel.setObjectName(u"knnFilePathLabel")

        self.formLayout.setWidget(10, QFormLayout.LabelRole, self.knnFilePathLabel)

        self.knnFilePathInput = QLineEdit(self.formLayoutWidget)
        self.knnFilePathInput.setObjectName(u"knnFilePathInput")

        self.formLayout.setWidget(10, QFormLayout.FieldRole, self.knnFilePathInput)

        self.startButton = QPushButton(self.centralwidget)
        self.startButton.setObjectName(u"startButton")
        self.startButton.setGeometry(QtCore.QRect(940, 640, 151, 28))
        self.startButton.clicked.connect(self.start_reading)
        self.backButton = QPushButton(self.centralwidget)
        self.backButton.setObjectName(u"backButton")
        self.backButton.setGeometry(QtCore.QRect(20, 640, 93, 28))
        self.backButton.clicked.connect(self.go_back)
        self.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(self)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1117, 26))
        self.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(self)
        self.statusbar.setObjectName(u"statusbar")
        self.setStatusBar(self.statusbar)

        self.inputs = [self.eegFrequencyInput, self.downsampleFrequencyInput, self.secondsPerEpochInput,
                       self.mouseIdInput, self.bufferLengthInput, self.rawTestDataFileInput, self.runNameInput,
                       self.trainingDataFileInput, self.trainingDataFileInput, self.ldaFilePathInput,
                       self.knnFilePathInput]

        for input in self.inputs:
            input.setText(self.settings.value(input.objectName()))

        set_input_default(self.eegFrequencyInput, Config.eeg_fs)
        set_input_default(self.downsampleFrequencyInput, Config.downsample_fs)
        set_input_default(self.secondsPerEpochInput, Config.num_seconds_per_epoch)
        set_input_default(self.bufferLengthInput, Config.median_filter_buffer)
        set_input_default(self.runNameInput, Config.run_name)

        self.retranslateUi()

    def go_back(self):
        self.goto("start")

    def start_reading(self):
        for input in self.inputs:
            self.settings.setValue(input.objectName(), input.text())

        #todo mg, come up with a better way to have two versions of the config
        config = SimpleNamespace(
            eeg_fs = int(self.eegFrequencyInputtext()),
            downsample_fs = int(self.downsampleFrequencyInputtext()),
            num_seconds_per_epoch = int(self.secondsPerEpochInputtext()),
            median_filter_buffer = int(self.bufferLengthInputtext()),
            median_filter_buffer_middle = math.ceil(int(self.bufferLengthInputtext()) / 2),
            mouse_ids = Config.mouse_ids,
            mouse_id_to_channel_mapping = Config.mouse_id_to_channel_mapping,
            print_timer_info_for_mice = Config.print_timer_info_for_mice,
            comport = Config.comport,
            cycle_test_data=Config.cycle_test_data,
            recreate_model_file = Config.recreate_model_file,
            recreate_lda = Config.recreate_lda,
            recreate_knn = Config.recreate_knn,
            base_path = Config.base_path,
            channel_file_base_path = Config.channel_file_base_path,
            data_path = Config.data_path,
            training_data_path = self.trainingDataFileInputtext(),
            raw_data_file = self.rawTestDataFileInputtext(),
            run_name = self.runNameInputtext(),
            multitaper_data_file_path = Config.multitaper_data_file_path,
            combined_data_file_path = self.trainingDataFileInputtext(),
            state_file_path = Config.state_file_path,
            lda_file_path = self.ldaFilePathInputtext(),
            knn_file_path = self.knnFilePathInputtext(),
            state_colors = Config.state_colors,
            get_channel_number_from_mouse_id = Config.get_channel_number_from_mouse_id
        )

        self.config_queue.put(config)

        self.goto("plot")

    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.eegFrequencyLabel.setText(QCoreApplication.translate("MainWindow", u"EEG frequency (ms)", None))
        self.downsampleFrequencyLabel.setText(
            QCoreApplication.translate("MainWindow", u"Downsample frequency (ms)", None))
        self.secondsLabel.setText(QCoreApplication.translate("MainWindow", u"Seconds per epoch", None))
        self.mouseIdLabel.setText(QCoreApplication.translate("MainWindow", u"Mouse IDs (comma seperated)", None))
        self.bufferLengthLabel.setText(QCoreApplication.translate("MainWindow", u"Buffer length", None))
        self.testDataLabel.setText(QCoreApplication.translate("MainWindow", u"Raw test data file", None))
        self.runNameLabel.setText(QCoreApplication.translate("MainWindow", u"Run name", None))
        self.trainingDataFileLabel.setText(QCoreApplication.translate("MainWindow", u"Training data file", None))
        self.trainingStatesFile.setText(QCoreApplication.translate("MainWindow", u"Training states file", None))
        self.ldaFilePathLabel.setText(QCoreApplication.translate("MainWindow", u"LDA file path", None))
        self.knnFilePathLabel.setText(QCoreApplication.translate("MainWindow", u"KNN file path", None))
        self.startButton.setText(QCoreApplication.translate("MainWindow", u"Start with these settings", None))
        self.backButton.setText(QCoreApplication.translate("MainWindow", u"< Back", None))
    # retranslateUi


#window for offline mode settings => OfflineSettingsWindow
class OfflineSettingsWindow(PageWindow):
    def __init__(self):
        super().__init__()


#window 6 for offline mode cycle of adjustment => ModelCreationWindow
class ModelCreationWindow(PageWindow):
    def __init__(self):
        super().__init__()

#window for now to just be the current online screen that we already have => PlotWindow
class PlotWindow(PageWindow):
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
        self.ax2.set_ylim([0, 80])

        self.figure3 = Figure(figsize=(5, 5))
        self.canvas3 = FigureCanvasQTAgg(self.figure3)
        self.ax3 = self.figure3.add_subplot(111, projection='3d')

        self.queue = queue
        self.mouse_id = Config.mouse_ids[0]#initialize to the first mouse number
        self.models = {str(num):MouseModel(num) for num in Config.mouse_ids}
        self.model = self.models[str(self.mouse_id)]
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
        layout.addWidget(self.canvas3, 0, 4, 6, 6)

        container.setLayout(layout)
        self.setCentralWidget(container)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.on_timeout)
        # timer interval in milliseconds
        timer.start(500)

    def mouse_change(self, mouse_id):
        self.mouse_id = mouse_id
        self.model = self.models[str(mouse_id)]
        self.ax1.clear()
        self.ax2.clear()
        #self.spot.remove()#broken

    def on_timeout(self):
        if not self.queue.empty():
            result = self.queue.get()

            class_name = result.standardized_class_name
            indicator_panel = self.indicator_panels[Config.mouse_ids.index(result.mouse_id)]
            indicator_panel.setText("Predicted class for mouse " + str(result.mouse_id) + " at timepoint" +
                                    str(result.time_point) + " :" + class_name)
            stylesheet = self.indicator_panel_stylesheet + "color: black; background-color: " \
                         + Config.state_colors[class_name]
            indicator_panel.setStyleSheet(stylesheet)

            if result.mouse_id is not self.mouse_id:
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
            self.ax2.plot([x for x in range(1, np.array(data).shape[1] + 1)], [point for point in data[0]])
            self.ax2.set_title("Transformed data")
            self.ax2.set_xlim([0, 200])
            self.ax2.set_ylim([0, 80])
            self.canvas2.draw()

            scatter_point = result.lda_point[0]
            self.spot.remove()
            self.spot = self.ax3.scatter(scatter_point[0], scatter_point[1], scatter_point[2], c='r', s=60)
            self.canvas3.draw()


def set_input_default(input, setting):
    text = input.text()
    if text is None or text == "":
        input.setText(str(setting))

def create_user_interface(input_queue, output_queue, config_queue):
    plot_app = QtWidgets.QApplication(sys.argv)
    p = Window(input_queue, config_queue)
    p.resize(1200, 892)
    p.show()
    plot_app.exec_()
    output_queue.put("Quit")