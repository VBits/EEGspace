"""
Online analysis
"""
import queue
import sys
from OnlineAnalysis import Config as ConfigOnline, InputProcessing, TestFileGeneration, StimulusOutput
from GUI import UserInterface
import multiprocessing
import threading
# from OfflineAnalysis import Config as OfflineConfig
# from OfflineAnalysis import ClusteringAndClassification

if __name__ == '__main__':

    args = sys.argv
    if '--run-offline-anaysis' in args:
        print("Analysing files in offline mode")
        # for mouse_id in OfflineConfig.mouse_id:
        #     ClusteringAndClassification.process_EEG_data()

    lock = threading.Lock()
    if ConfigOnline.cycle_test_data:
        TestFileGeneration.cycle_test_files(lock)
    jobs = []
    manager = multiprocessing.Manager()
    file_queue = manager.Queue()
    ui_input_queue = manager.Queue()
    ui_output_queue = manager.Queue()
    # stimulus_queues = []

    p = multiprocessing.Process(target=UserInterface.create_user_interface, args=(ui_input_queue, ui_output_queue))
    p.daemon = True
    p.start()
    jobs.append(p)

    #start processing EEG data
    def run_loop_processes(config):
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

