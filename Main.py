import queue
import sys
import Config
import multiprocessing
import TestFileGeneration
import SimpleUserInterface
import InputProcessing
import threading
import StimulusOutput

if __name__ == '__main__':
    lock = threading.Lock()
    if Config.cycle_test_data:
        TestFileGeneration.cycle_test_files(lock)
    jobs = []
    manager = multiprocessing.Manager()
    file_queue = manager.Queue()
    ui_input_queue = manager.Queue()
    ui_output_queue = manager.Queue()
    # stimulus_queues = []

    p = multiprocessing.Process(target=SimpleUserInterface.create_user_interface,
                                args=(ui_input_queue, ui_output_queue, Config.mice_numbers))
    jobs.append(p)
    p.start()

    for mouse_number in Config.mice_numbers:
        p = multiprocessing.Process(target=InputProcessing.run_loop, args=(mouse_number, file_queue))
        jobs.append(p)
        p.start()
        sys.stdout.flush()
        #for later when we have other forms of stimulus
        # stimulus_input_queue = queue.Queue()
        # stimulus_output_queue = queue.Queue()
        # stimulus_queues[mouse_number - 1] = queue
        # threading.Thread(target=StimulusOutput.randomize_stimulus_output,
        #                  args=(stimulus_input_queue, stimulus_output_queue, "BrainLaser"))

    stimulus_input_queue = queue.Queue()
    stimulus_output_queue = queue.Queue()
    threading.Thread(target=StimulusOutput.randomize_stimulus_output,
                     args=(stimulus_input_queue, stimulus_output_queue, "WhiteNoise"))

    map(lambda p: p.join(), jobs)

    while True:
        if not ui_output_queue.empty():
            output = ui_output_queue.get()
            if output == "Quit":
                break
            #todo kill test file cycling as well
        while not file_queue.empty():
            next_status = file_queue.get()
            #stimulus_queues[next_status.mouse_number - 1].put(next_status)
            stimulus_input_queue.put(next_status)
            ui_input_queue.put(next_status)
            print(next_status)
