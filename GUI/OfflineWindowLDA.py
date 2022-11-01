# window for offline mode settings => OfflineSettingsWindow
class OfflineWindowLDA(PageWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings("ClosedLoopEEG", "LDA")
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

        self.inputs = [self.basePathInput, self.experimentalPathInput, self.fileInput, self.mouseDescriptionInput,
                       self.mouseIdInput, self.experimentIdInput, self.fileIdInput, self.dpaZInput,
                       self.targetFsInput,
                       self.epochSecondsInput, self.smoothingWindowInput, self.randomEpochSizeInput,
                       self.ldaComponentsInput, self.dpaKMaxInput, self.knnNNeighborsInput]

        for input in self.inputs:
            input.setText(self.settings.value(input.objectName()))

        set_input_default(self.basePathInput, OfflineConfig.base_path)
        set_input_default(self.experimentalPathInput, OfflineConfig.experimental_path)
        set_input_default(self.basePathInput, OfflineConfig.base_path)
        set_input_default(self.fileInput, OfflineConfig.file)
        set_input_default(self.mouseDescriptionInput, OfflineConfig.mouse_description)
        set_input_default(self.mouseIdInput, OfflineConfig.mouse_id)
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

    def start_reading(self):
        for input in self.inputs:
            self.settings.setValue(input.objectName(), input.text())

        self.goto("plot")

    def retranslateUi(self):
        self.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.basePathLabel.setText(QCoreApplication.translate("MainWindow", u"Base Path", None))
        self.experimentalPathLabel.setText(
            QCoreApplication.translate("MainWindow", u"Experimental Path", None))
        self.fileLabel.setText(QCoreApplication.translate("MainWindow", u"File", None))
        self.mouseDescriptionLabel.setText(
            QCoreApplication.translate("MainWindow", u"Mouse Description (strain, mutations etc)", None))
        self.mouseIdLabel.setText(QCoreApplication.translate("MainWindow", u"Mouse ID (position in rig)", None))
        self.experimentIdLabel.setText(QCoreApplication.translate("MainWindow", u"Experiment ID", None))
        self.fileIdLabel.setText(QCoreApplication.translate("MainWindow", u"File ID", None))
        self.dpaZLabel.setText(QCoreApplication.translate("MainWindow", u"DPA Z", None))
        self.targetFsLabel.setText(QCoreApplication.translate("MainWindow", u"Target Fs", None))
        self.epochSecondsLabel.setText(
            QCoreApplication.translate("MainWindow", u"Epoch Duration (in seconds)", None))
        self.smoothingWindowLabel.setText(
            QCoreApplication.translate("MainWindow", u"Smoothing window (# of epochs)", None))
        self.randomEpochSizeLabel.setText(
            QCoreApplication.translate("MainWindow", u"Numbers of random epochs to process", None))
        self.ldaComponentsLabel.setText(QCoreApplication.translate("MainWindow", u"LDA components", None))
        self.dpaKMaxLabel.setText(QCoreApplication.translate("MainWindow", u"DPA k max", None))
        self.knnNNeighborsLabel.setText(QCoreApplication.translate("MainWindow", u"Numbers of knn neighbors", None))
        self.startButton.setText(QCoreApplication.translate("MainWindow", u"Start with these settings", None))
        self.backButton.setText(QCoreApplication.translate("MainWindow", u"< Back", None))

    def offline_plots(self):
        self.setWindowTitle("Offline analysis EEG")

        self.figure1 = Figure(figsize=(5, 5))
        self.canvas1 = FigureCanvasQTAgg(self.figure1)
        self.ax1 = self.figure1.add_subplot(111)
        self.ax1.set_title("Raw data")
        # self.ax1.set_xlim([0, 400])
        # self.ax1.set_ylim([-300, 300])

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
        self.mouse_id = OnlineConfig.mouse_ids[0]  # initialize to the first mouse number
        self.models = {str(num): MouseModel(num) for num in OnlineConfig.mouse_ids}
        self.model = self.models[str(self.mouse_id)]
        self.lda_encoded = self.model.lda.transform(self.model.training_data)

    def offline_plots(self):
        self.setWindowTitle("Offline analysis EEG")

        self.figure1 = Figure(figsize=(5, 5))
        self.canvas1 = FigureCanvasQTAgg(self.figure1)
        self.ax1 = self.figure1.add_subplot(111)
        self.ax1.set_title("Raw data")
        # self.ax1.set_xlim([0, 400])
        # self.ax1.set_ylim([-300, 300])

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
        self.mouse_id = OnlineConfig.mouse_ids[0]  # initialize to the first mouse number
        self.models = {str(num): MouseModel(num) for num in OnlineConfig.mouse_ids}
        self.model = self.models[str(self.mouse_id)]
        self.lda_encoded = self.model.lda.transform(self.model.training_data)