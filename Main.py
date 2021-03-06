"""
Online analysis
"""
import queue
import sys
from OnlineAnalysis import Config as ConfigOnline, InputProcessing, TestFileGeneration, StimulusOutput, UserInterface
import multiprocessing
import threading

if __name__ == '__main__':

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
    jobs.append(p)
    p.start()

    #start processing EEG data
    for mouse_number in ConfigOnline.rig_position:
        p = multiprocessing.Process(target=InputProcessing.run_loop, args=(mouse_number, file_queue))
        p.daemon = True
        jobs.append(p)
        p.start()
        sys.stdout.flush()
        #TODO change only if you're delivering stimuli to mice at diffirent times
        #for later when we have other forms of stimulus
        # stimulus_input_queue = queue.Queue()
        # stimulus_output_queue = queue.Queue()
        # stimulus_queues[mouse_number - 1] = queue
        # threading.Thread(target=StimulusOutput.randomize_stimulus_output,
        #                  args=(stimulus_input_queue, stimulus_output_queue, "BrainLaser"))

    #Bulk stimulus
    # stimulus_input_queue = queue.Queue()
    # stimulus_output_queue = queue.Queue()
    # stimulus_thread = threading.Thread(target=StimulusOutput.randomize_stimulus_output,
    #                                    args=(stimulus_input_queue, stimulus_output_queue, "WhiteNoise"))
    # stimulus_thread.start()
    # stimulus_thread.join()

    map(lambda p: p.join(), jobs)

    while True:
        if not ui_output_queue.empty():
            output = ui_output_queue.get()
            if output == "Quit":
                for p in jobs:
                    p.terminate()
                break
        while not file_queue.empty():
            next_status = file_queue.get()
            #stimulus_queues[next_status.mouse_number - 1].put(next_status)
            stimulus_input_queue.put(next_status)
            ui_input_queue.put(next_status)
            print(next_status)
