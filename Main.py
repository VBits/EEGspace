import threading
import Config
import TestFileGeneration
import multiprocessing
import sys
import SimpleUserInterface
import InputProcessing


if __name__ == '__main__':
    lock = threading.Lock()
    if Config.cycle_test_data:
        TestFileGeneration.cycle_test_files(lock)
    jobs = []
    manager = multiprocessing.Manager()
    file_queue = manager.Queue()
    ui_input_queue = manager.Queue()
    ui_output_queue = manager.Queue()

    p = multiprocessing.Process(target=SimpleUserInterface.create_user_interface,
                                args=(ui_input_queue, ui_output_queue, Config.mice_numbers))
    jobs.append(p)
    p.start()

    for i in Config.mice_numbers:
        p = multiprocessing.Process(target=InputProcessing.run_loop, args=(i, file_queue))
        jobs.append(p)
        p.start()
        sys.stdout.flush()

    map(lambda p: p.join(), jobs)

    while True:
        if not ui_output_queue.empty():
            output = ui_output_queue.get()
            if output == "Quit":
                break
            #todo kill test file cycling as well
        while not file_queue.empty():
            next_status = file_queue.get()
            ui_input_queue.put(next_status)
            print(next_status)
