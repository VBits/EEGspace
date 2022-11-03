from PyQt5 import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

class PageWindow(QtWidgets.QMainWindow):
    gotoSignal = QtCore.pyqtSignal(str)
    updateParentStateSignal = QtCore.pyqtSignal(object)
    state = {}

    def goto(self, name):
        self.gotoSignal.emit(name)

    def updateParentState(self, state):
        self.updateParentStateSignal.emit(state)

    @QtCore.pyqtSlot(object)
    def setState(self, state):
        self.state = state

    def mount(self):
        pass
