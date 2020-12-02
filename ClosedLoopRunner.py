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


def run_loop(channel_number, file_lock, model):
    #for testing convenience
    #annoying, remove later
    # if Config.cycle_test_data:
    #     f = open(Config.training_data_path + "Sxx_norm_200604_m1.pkl", 'rb')
    #     eeg_data = pickle.load(f)
    #     epoch_count = 41
    #     data_points = [x for x in np.array(eeg_data[0:41])]
    # else:
    epoch_count = 0
    data_points = []
    time_points = []
    total_points = 0
    path = Config.channel_file_base_path.format(channel_number=channel_number)
    iteration = 0
    timer = Timer("start_time", channel_number, iteration)
    time_for_iterations = []
    while True:
        if not os.path.isfile(path):
            time.sleep(0.30)
            continue
        timer.print_duration_since("start_time", "Time waiting for file")
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
            print("Predicted class for mouse " + str(channel_number) + " is " + model.states[predicted_class[0]])
            data_points = data_points[1:]

            timer.print_duration_since("start_data_analysis", "Time doing data analysis")

        epoch_count = epoch_count + 1
        timer.print_duration_since("start_time", "Time for iteration was")
        #todo remove
        # time_for_iterations.append(timer.get_duration_since("start_time"))
        # print(time_for_iterations)
        # print("Average time for iterations for mouse " + str(channel_number) + ": " + str(np.mean(np.array(time_for_iterations))))
        iteration = iteration + 1
        timer = Timer("start_time", channel_number, iteration)


lock = threading.Lock()
if Config.cycle_test_data:
    CycleTestData.cycle_test_files(lock)
cycle_files(lock)
