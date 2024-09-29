"""
Online analysis
"""
from PyQt5 import QtGui, QtCore, QtWidgets

import sys
from OnlineAnalysis import Config as ConfigOnline
from GUI import UserInterface
import multiprocessing
import threading

def run_test_file_cycle():
    from OnlineAnalysis import TestFileGeneration
    TestFileGeneration.cycle_test_files(lock)

# from OfflineAnalysis import Config as OfflineConfig
# from OfflineAnalysis import ClusteringAndClassification

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    # Create splashscreen, this doesn't currently work
    splash_pix = QtGui.QPixmap('img/Loading_icon.gif')
    splash = QtWidgets.QSplashScreen(splash_pix, QtCore.Qt.WindowStaysOnTopHint)
    splash.show()
    # start up the multiprocessing (probably need to run all the calculations and stuff seperate to the UI thread
    with multiprocessing.Manager() as manager:

        args = sys.argv
        # if '--run-offline-anaysis' in args:
        #     print("Analysing files in offline mode")
        #     # for mouse_id in OfflineConfig.mouse_id:
        #     #     ClusteringAndClassification.process_EEG_data()

        if ConfigOnline.cycle_test_data:
            lock = threading.Lock()
            run_test_file_cycle()
        # print("time since start 0.1: ", timer.get_duration_since("start_time"))
        jobs = []
        #manager = multiprocessing.Manager()
        # print("time since start 0.2: ", timer.get_duration_since("start_time"))
        file_queue = manager.Queue()
        # print("time since start 0.3: ", timer.get_duration_since("start_time"))
        ui_input_queue = manager.Queue()
        # print("time since start 0.4: ", timer.get_duration_since("start_time"))
        ui_output_queue = manager.Queue()
        # print("time since start 1: ", timer.get_duration_since("start_time"))
        stimulus_queues = []

        UserInterface.create_user_interface(ui_input_queue, ui_output_queue)
        # p = multiprocessing.Process(target=UserInterface.create_user_interface, args=(ui_input_queue, ui_output_queue))
        # p.daemon = True
        # p.start()
        # jobs.append(p)
        #print("time since start: 2", timer.get_duration_since("start_time"))

        #start processing EEG data
        def run_loop_processes(config):
            from OnlineAnalysis import InputProcessing  # this takes 2.8 seconds or so
            for mouse_id in config.mouse_ids:
                p = multiprocessing.Process(target=InputProcessing.run_loop, args=(mouse_id, file_queue, LoadModels, config))
                p.daemon = True
                p.start()
                jobs.append(p)
                sys.stdout.flush()
                #TODO change only if you're delivering stimuli to mice at diffirent times
                #for later when we have other forms of stimulus
                # stimulus_input_queue = queue.Queue()
                # stimulus_output_queue = queue.Queue()
                # stimulus_queues[mouse_id - 1] = queue
                # threading.Thread(target=StimulusOutput.randomize_stimulus_output,
                #                  args=(stimulus_input_queue, stimulus_output_queue, "BrainLaser"))

        #Bulk stimulus
        # stimulus_input_queue = queue.Queue()
        # stimulus_output_queue = queue.Queue()
        # stimulus_thread = threading.Thread(target=StimulusOutput.randomize_stimulus_output,
        #                                    args=(stimulus_input_queue, stimulus_output_queue, "WhiteNoise"))
        # stimulus_thread.start()
        # stimulus_thread.join()

        while True:
            if not ui_output_queue.empty():
                output = ui_output_queue.get()
                if output == "Quit":
                    sys.exit(1)
            while not file_queue.empty():
                next_status = file_queue.get()
                #stimulus_queues[next_status.mouse_id - 1].put(next_status)
                #stimulus_input_queue.put(next_status)
                ui_input_queue.put(next_status)
                print(next_status)

