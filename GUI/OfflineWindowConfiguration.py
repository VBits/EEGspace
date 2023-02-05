from OnlineAnalysis.Timing import Timer
timer = Timer("start_time", None, None)
print("time since start c0: ", timer.get_duration_since("start_time"))
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
print("time since start c1: ", timer.get_duration_since("start_time"))

from GUI.Utilities import set_input_default
from GUI.PageWindow import PageWindow
from OfflineAnalysis import Config as OfflineConfig
print("time since start c1.1: ", timer.get_duration_since("start_time"))
from OfflineAnalysis.Utilities.LoadData import process_EEG_data
print("time since start c2: ", timer.get_duration_since("start_time"))
import traceback
import sys
print("time since start c3: ", timer.get_duration_since("start_time"))

#window for offline mode settings => OfflineSettingsWindow
class OfflineSettingsWindow(PageWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("ClosedLoopEEG", "OfflineSettings")
        self.resize(1117, 892)
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName(u"centralwidget")
        self.formLayoutWidget = QWidget(self.centralwidget)
        self.formLayoutWidget.setObjectName(u"formLayoutWidget")
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 30, 1071, 681))
        self.formLayout = QFormLayout(self.formLayoutWidget)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setContentsMargins(0, 0, 0, 0)


        self.basePathLabel = QLabel(self.formLayoutWidget)
        self.basePathLabel.setObjectName(u"basePathLabel")
        self.formLayout.setWidget(0, QFormLayout.LabelRole, self.basePathLabel)

        self.basePathInput = QLineEdit(self.formLayoutWidget)
        self.basePathInput.setObjectName(u"basePathInput")
        self.formLayout.setWidget(0, QFormLayout.FieldRole, self.basePathInput)

        self.experimentalPathLabel = QLabel(self.formLayoutWidget)
        self.experimentalPathLabel.setObjectName(u"experimentalPathLabel")
        self.formLayout.setWidget(1, QFormLayout.LabelRole, self.experimentalPathLabel)

        self.experimentalPathInput = QLineEdit(self.formLayoutWidget)
        self.experimentalPathInput.setObjectName(u"experimentalPathInput")

        self.formLayout.setWidget(1, QFormLayout.FieldRole, self.experimentalPathInput)

        self.fileLabel = QLabel(self.formLayoutWidget)
        self.fileLabel.setObjectName(u"fileLabel")

        self.formLayout.setWidget(2, QFormLayout.LabelRole, self.fileLabel)

        self.fileInput = QLineEdit(self.formLayoutWidget)
        self.fileInput.setObjectName(u"fileInput")

        self.formLayout.setWidget(2, QFormLayout.FieldRole, self.fileInput)

        self.mouseDescriptionLabel = QLabel(self.formLayoutWidget)
        self.mouseDescriptionLabel.setObjectName(u"mouseDescriptionLabel")

        self.formLayout.setWidget(3, QFormLayout.LabelRole, self.mouseDescriptionLabel)

        self.mouseDescriptionInput = QLineEdit(self.formLayoutWidget)
        self.mouseDescriptionInput.setObjectName(u"mouseDescriptionInput")

        self.formLayout.setWidget(3, QFormLayout.FieldRole, self.mouseDescriptionInput)

        self.mouseIdLabel = QLabel(self.formLayoutWidget)
        self.mouseIdLabel.setObjectName(u"mouseIdLabel")

        self.formLayout.setWidget(4, QFormLayout.LabelRole, self.mouseIdLabel)

        self.mouseIdInput = QLineEdit(self.formLayoutWidget)
        self.mouseIdInput.setObjectName(u"mouseIdInput")

        self.formLayout.setWidget(4, QFormLayout.FieldRole, self.mouseIdInput)

        self.experimentIdLabel = QLabel(self.formLayoutWidget)
        self.experimentIdLabel.setObjectName(u"experimentIdLabel")

        self.formLayout.setWidget(5, QFormLayout.LabelRole, self.experimentIdLabel)

        self.experimentIdInput = QLineEdit(self.formLayoutWidget)
        self.experimentIdInput.setObjectName(u"experimentIdInput")

        self.formLayout.setWidget(5, QFormLayout.FieldRole, self.experimentIdInput)

        self.fileIdLabel = QLabel(self.formLayoutWidget)
        self.fileIdLabel.setObjectName(u"fileIdLabel")

        self.formLayout.setWidget(6, QFormLayout.LabelRole, self.fileIdLabel)

        self.fileIdInput = QLineEdit(self.formLayoutWidget)
        self.fileIdInput.setObjectName(u"fileIdInput")

        self.formLayout.setWidget(6, QFormLayout.FieldRole, self.fileIdInput)

        self.dpaZLabel = QLabel(self.formLayoutWidget)
        self.dpaZLabel.setObjectName(u"dpaZLabel")

        self.formLayout.setWidget(7, QFormLayout.LabelRole, self.dpaZLabel)

        self.dpaZInput = QLineEdit(self.formLayoutWidget)
        self.dpaZInput.setObjectName(u"dpaZInput")

        self.formLayout.setWidget(7, QFormLayout.FieldRole, self.dpaZInput)

        self.targetFsLabel = QLabel(self.formLayoutWidget)
        self.targetFsLabel.setObjectName(u"targetFsLabel")

        self.formLayout.setWidget(8, QFormLayout.LabelRole, self.targetFsLabel)

        self.targetFsInput = QLineEdit(self.formLayoutWidget)
        self.targetFsInput.setObjectName(u"targetFsInput")

        self.formLayout.setWidget(8, QFormLayout.FieldRole, self.targetFsInput)

        self.epochSecondsLabel = QLabel(self.formLayoutWidget)
        self.epochSecondsLabel.setObjectName(u"epochSecondsLabel")

        self.formLayout.setWidget(9, QFormLayout.LabelRole, self.epochSecondsLabel)

        self.epochSecondsInput = QLineEdit(self.formLayoutWidget)
        self.epochSecondsInput.setObjectName(u"epochSecondsInput")

        self.formLayout.setWidget(9, QFormLayout.FieldRole, self.epochSecondsInput)

        self.smoothingWindowLabel = QLabel(self.formLayoutWidget)
        self.smoothingWindowLabel.setObjectName(u"smoothingWindowLabel")

        self.formLayout.setWidget(10, QFormLayout.LabelRole, self.smoothingWindowLabel)

        self.smoothingWindowInput = QLineEdit(self.formLayoutWidget)
        self.smoothingWindowInput.setObjectName(u"smoothingWindowInput")

        self.formLayout.setWidget(10, QFormLayout.FieldRole, self.smoothingWindowInput)

        self.randomEpochSizeLabel = QLabel(self.formLayoutWidget)
        self.randomEpochSizeLabel.setObjectName(u"randomEpochSizeLabel")

        self.formLayout.setWidget(11, QFormLayout.LabelRole, self.randomEpochSizeLabel)

        self.randomEpochSizeInput = QLineEdit(self.formLayoutWidget)
        self.randomEpochSizeInput.setObjectName(u"randomEpochSizeInput")

        self.formLayout.setWidget(11, QFormLayout.FieldRole, self.randomEpochSizeInput)

        self.ldaComponentsLabel = QLabel(self.formLayoutWidget)
        self.ldaComponentsLabel.setObjectName(u"ldaComponentsLabel")

        self.formLayout.setWidget(12, QFormLayout.LabelRole, self.ldaComponentsLabel)

        self.ldaComponentsInput = QLineEdit(self.formLayoutWidget)
        self.ldaComponentsInput.setObjectName(u"ldaComponentsInput")

        self.formLayout.setWidget(12, QFormLayout.FieldRole, self.ldaComponentsInput)

        self.dpaKMaxLabel = QLabel(self.formLayoutWidget)
        self.dpaKMaxLabel.setObjectName(u"dpaKMaxLabel")

        self.formLayout.setWidget(13, QFormLayout.LabelRole, self.dpaKMaxLabel)

        self.dpaKMaxInput = QLineEdit(self.formLayoutWidget)
        self.dpaKMaxInput.setObjectName(u"dpaKMaxInput")

        self.formLayout.setWidget(13, QFormLayout.FieldRole, self.dpaKMaxInput)

        self.knnNNeighborsLabel = QLabel(self.formLayoutWidget)
        self.knnNNeighborsLabel.setObjectName(u"knnNNeighborsLabel")

        self.formLayout.setWidget(14, QFormLayout.LabelRole, self.knnNNeighborsLabel)

        self.knnNNeighborsInput = QLineEdit(self.formLayoutWidget)
        self.knnNNeighborsInput.setObjectName(u"knnNNeighborsInput")

        self.formLayout.setWidget(14, QFormLayout.FieldRole, self.knnNNeighborsInput)

        self.sxxOriginLabel = QLabel(self.formLayoutWidget)
        self.sxxOriginLabel.setText("Load Sxx?")

        self.formLayout.setWidget(15, QFormLayout.LabelRole, self.sxxOriginLabel)

        self.sxxOrigin = QComboBox()
        self.sxxOrigin.addItems(["Preprocess raw data", "Load previously previously stored .pkl files"])

        self.formLayout.setWidget(15, QFormLayout.FieldRole, self.sxxOrigin)

        self.statesOriginLabel = QLabel(self.formLayoutWidget)
        self.statesOriginLabel.setText("Load states?")

        self.formLayout.setWidget(16, QFormLayout.LabelRole, self.statesOriginLabel)

        self.statesOrigin = QComboBox()
        self.statesOrigin.addItems(["Don't load m.state_df", "Load m.state_df"])

        self.formLayout.setWidget(16, QFormLayout.FieldRole, self.statesOrigin)

        self.startButton = QPushButton(self.centralwidget)
        self.startButton.setObjectName(u"startButton")
        self.startButton.setGeometry(QtCore.QRect(940, 640, 151, 28))
        self.startButton.clicked.connect(self.go_to_next)
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

        self.inputs = [self.basePathInput, self.experimentalPathInput, self.fileInput, self.mouseDescriptionInput,
                       self.mouseIdInput, self.experimentIdInput, self.fileIdInput, self.dpaZInput, self.targetFsInput,
                       self.epochSecondsInput, self.smoothingWindowInput, self.randomEpochSizeInput,
                       self.ldaComponentsInput, self.dpaKMaxInput, self.knnNNeighborsInput]


        for input in self.inputs:
            input.setText(self.settings.value(input.objectName()))

        set_input_default(self.basePathInput, OfflineConfig.base_path)
        set_input_default(self.experimentalPathInput, OfflineConfig.experimental_path)
        set_input_default(self.basePathInput, OfflineConfig.base_path)
        set_input_default(self.fileInput, OfflineConfig.file)
        set_input_default(self.mouseDescriptionInput, OfflineConfig.mouse_description)
        set_input_default(self.mouseIdInput,OfflineConfig.mouse_id)
        set_input_default(self.experimentIdInput, OfflineConfig.experiment_id)
        set_input_default(self.fileIdInput, OfflineConfig.file_id)
        set_input_default(self.dpaZInput, OfflineConfig.dpa_z)
        set_input_default(self.targetFsInput, OfflineConfig.target_fs)
        set_input_default(self.epochSecondsInput, OfflineConfig.epoch_seconds)
        set_input_default(self.smoothingWindowInput, OfflineConfig.smoothing_window)
        set_input_default(self.randomEpochSizeInput, OfflineConfig.random_epoch_size)
        set_input_default(self.ldaComponentsInput, OfflineConfig.lda_components)
        set_input_default(self.dpaKMaxInput, OfflineConfig.dpa_k_max)
        set_input_default(self.knnNNeighborsInput, OfflineConfig.knn_n_neighbors)

        self.retranslateUi()

    def go_back(self):
        self.goto("start")

    def go_to_next(self):
        self.goto("lda_window")
        return

        for input in self.inputs:
            self.settings.setValue(input.objectName(), input.text())

        try:
            load_data = self.sxxOrigin.currentIndex() + 1
            print(load_data, "load data index")
            mouse = process_EEG_data(self.mouseDescriptionInput.text(), int(self.mouseIdInput.text()), load_data)
            print("got here")
            state = { k:v for k,v in self.state.items() }
            state["mouse"] = mouse
            print("New state", state)
            self.updateParentState(state)
            self.goto("lda_window")
        except Exception as e:
            traceback.print_exception(*sys.exc_info())
            print(e)

    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.basePathLabel.setText(QCoreApplication.translate("MainWindow", u"Base Path", None))
        self.experimentalPathLabel.setText(
            QCoreApplication.translate("MainWindow", u"Experimental Path", None))
        self.fileLabel.setText(QCoreApplication.translate("MainWindow", u"File", None))
        self.mouseDescriptionLabel.setText(QCoreApplication.translate("MainWindow", u"Mouse Description (strain, mutations etc)", None))
        self.mouseIdLabel.setText(QCoreApplication.translate("MainWindow", u"Mouse ID (position in rig)", None))
        self.experimentIdLabel.setText(QCoreApplication.translate("MainWindow", u"Experiment ID", None))
        self.fileIdLabel.setText(QCoreApplication.translate("MainWindow", u"File ID", None))
        self.dpaZLabel.setText(QCoreApplication.translate("MainWindow", u"DPA Z", None))
        self.targetFsLabel.setText(QCoreApplication.translate("MainWindow", u"Target Fs", None))
        self.epochSecondsLabel.setText(QCoreApplication.translate("MainWindow", u"Epoch Duration (in seconds)", None))
        self.smoothingWindowLabel.setText(QCoreApplication.translate("MainWindow", u"Smoothing window (# of epochs)", None))
        self.randomEpochSizeLabel.setText(QCoreApplication.translate("MainWindow", u"Numbers of random epochs to process", None))
        self.ldaComponentsLabel.setText(QCoreApplication.translate("MainWindow", u"LDA components", None))
        self.dpaKMaxLabel.setText(QCoreApplication.translate("MainWindow", u"DPA k max", None))
        self.knnNNeighborsLabel.setText(QCoreApplication.translate("MainWindow", u"Numbers of knn neighbors", None))
        self.startButton.setText(QCoreApplication.translate("MainWindow", u"Start with these settings", None))
        self.backButton.setText(QCoreApplication.translate("MainWindow", u"< Back", None))