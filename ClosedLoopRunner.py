import pandas
import struct
import numpy as np
import threading
import os
import Config
import time
from Model import get_model_object
import CycleTestData
import Preprocessing
import pickle
from Timer import Timer
import multiprocessing
import sys
from multiprocessing import Queue
import UserInterface


def cycle_files(file_lock):
    jobs = []

    for i in Config.mice_numbers:
        p = multiprocessing.Process(target=run_loop, args=(i,))
        jobs.append(p)
        p.start()

    map(lambda p: p.join(), jobs)


def read_next_epoch_from_file():
    with file_lock:
        with open(path, "rb") as f:
            bytes_read = f.read(8)
            time_points.append(struct.unpack('<d', bytes_read)[0])
            bytes_read = f.read(4)
            total_points = total_points + struct.unpack('<i', bytes_read)[0]
            bytes_read = f.read(4)
            while bytes_read:
                data_points.append(struct.unpack('<f', bytes_read)[0])
                bytes_read = f.read(4)
        os.remove(path)

def run_loop(mouse_number, queue):
    sys.stdout.flush()
    file_lock = threading.Lock()#todo try this with multiprocessing, is it needed for cycling test files? I think it is
    model = get_model_object(mouse_number)
    epoch_count = 0
    data_points = []
    time_points = []
    total_points = 0
    path = Config.channel_file_base_path.format(channel_number=(mouse_number - 1))#zero indexed
    iteration = 0
    timer = Timer("start_time", mouse_number, iteration)
    dropped_epochs = []
    while not os.path.isfile(path):
        continue
    sys.stdout.flush()
    #record the timestamp before the first run and run it every 2 seconds after that
    seconds_per_iteration = Config.num_seconds_per_epoch

    # file has been found, wait for it to be read
    time.sleep(0.30)
    # remove the first file
    os.remove(path)

    iteration_deadline = time.time() + seconds_per_iteration

    while True:
        #sys.stdout.
        if time.time() > iteration_deadline:
            if mouse_number in Config.print_timer_info_for_mice:
                print(time.time())
                print(iteration_deadline)
            if os.path.isfile(path):
                os.remove(path)
            num_epochs_dropped = int((time.time() - iteration_deadline) / seconds_per_iteration) + 1
            dropped_epochs_delay_seconds = num_epochs_dropped * seconds_per_iteration
            dropped_epochs = dropped_epochs + list(range(epoch_count, epoch_count + num_epochs_dropped))
            epoch_count += num_epochs_dropped - 1
            iteration += num_epochs_dropped - 1
            iteration_deadline += dropped_epochs_delay_seconds

        while time.time() < iteration_deadline:
            continue

        iteration_deadline += seconds_per_iteration
        timer.print_duration_since("start_time", "Time for iteration was")
        iteration += 1
        timer = Timer("start_time", mouse_number, iteration)

        timer.set_time_point("start_reading_file")
        # next_epoch = read_next_epoch_from_file(file_lock)
        #data_points.append(next_epoch)
        with file_lock:
            with open(path, "rb") as f:
                bytes_read = f.read(8)
                time_points.append(struct.unpack('<d', bytes_read)[0])
                bytes_read = f.read(4)
                total_points = total_points + struct.unpack('<i', bytes_read)[0]
                print(total_points)
                bytes_read = f.read(4)
                while bytes_read:
                    data_points.append(struct.unpack('<f', bytes_read)[0])
                    bytes_read = f.read(4)
            os.remove(path)
        print("total points for mouse " + str(mouse_number) + " is " + str(total_points))
        timer.print_duration_since("start_reading_file", "Time doing file reading")

        if epoch_count > Config.iteration_buffer:
            timer.set_time_point("start_data_analysis")
            transformed_data = Preprocessing.transform_data(data_points, timer, model.norm)
            # import matplotlib.pyplot as plt
            # plt.plot(transformed_data[50])
            # plt.show()
            transformed_data = [np.array(transformed_data)[-1]]
            lda_point = model.lda.transform(transformed_data)
            predicted_class = model.classifier.predict(lda_point)
            original_class_number = predicted_class[0]
            standardized_class_number = model.state_mappings[original_class_number]
            standardized_class_name = model.get_standard_state_name(standardized_class_number)
            queue.put((mouse_number, epoch_count, standardized_class_number,
                       standardized_class_name, original_class_number))
            data_points = data_points[Config.eeg_fs * Config.num_seconds_per_epoch:]
            if mouse_number in Config.print_timer_info_for_mice:
                print(len(data_points))
                print(dropped_epochs)
            timer.print_duration_since("start_data_analysis", "Time doing data analysis")

        epoch_count = epoch_count + 1
        sys.stdout.flush()


if __name__ == '__main__':
    lock = threading.Lock()
    if Config.cycle_test_data:
        CycleTestData.cycle_test_files(lock)
    jobs = []
    manager = multiprocessing.Manager()
    file_queue = manager.Queue()
    ui_input_queue = manager.Queue()
    ui_output_queue = manager.Queue()

    p = multiprocessing.Process(target=UserInterface.create_user_interface,
                                args=(ui_input_queue, ui_output_queue, Config.mice_numbers))
    jobs.append(p)
    p.start()

    for i in Config.mice_numbers:
        p = multiprocessing.Process(target=run_loop, args=(i, file_queue))
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
