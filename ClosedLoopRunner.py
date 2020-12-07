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


def cycle_files(file_lock):
    for i in Config.mice_numbers:
        #run_loop(i, file_lock, training_mouse_object)
        model = get_model_object(i)
        threading.Thread(target=run_loop, args=(i, file_lock, model)).start()


def run_loop(mouse_number, file_lock, model):
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

    #record the timestamp before the first run and run it every 2 seconds after that
    seconds_per_iteration = Config.num_seconds_per_epoch

    # file has been found, wait for it to be read
    time.sleep(0.30)
    # remove the first file
    os.remove(path)

    iteration_deadline = time.time() + seconds_per_iteration

    while True:
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
        timer.print_duration_since("start_reading_file", "Time doing file reading")

        if epoch_count > Config.iteration_buffer:
            timer.set_time_point("start_data_analysis")
            point = model.lda.transform(Preprocessing.transform_data(data_points, timer, model.norm))
            predicted_class = model.classifier.predict(point)
            print("Predicted class for mouse " + str(mouse_number) + " is " + model.states[predicted_class[0]])
            data_points = data_points[1:]
            if mouse_number in Config.print_timer_info_for_mice:
                print(dropped_epochs)
            timer.print_duration_since("start_data_analysis", "Time doing data analysis")

        epoch_count = epoch_count + 1



lock = threading.Lock()
if Config.cycle_test_data:
    CycleTestData.cycle_test_files(lock)
cycle_files(lock)
