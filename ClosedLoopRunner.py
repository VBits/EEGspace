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


def cycle_files(file_lock):
    model = get_model_object()
    for i in range(0, Config.num_channels):
        #run_loop(i, file_lock, training_mouse_object)
        threading.Thread(target=run_loop, args=(i, file_lock, model)).start()


def run_loop(channel_number, file_lock, model):
    epoch_count = 0
    data_points = []
    time_points = []
    total_points = 0
    start_time = time.perf_counter()
    path = Config.channel_file_base_path.format(channel_number=channel_number)
    while True:
        if not os.path.isfile(path):
            time.sleep(0.30)
            continue
        end_polling_for_file = time.perf_counter()
        print("time waiting for file: " + str(end_polling_for_file - start_time))
        start_reading_file = time.perf_counter()
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
        end_reading_file = time.perf_counter()
        print("time doing file reading: " + str(end_reading_file - start_reading_file))

        if epoch_count > 41:
            start_data_analysis = time.perf_counter()
            X = model.lda.transform(Preprocessing.transform_data(data_points))
            point = X[-1]
            predicted_class = model.classifier.predict(point)
            print("Predicted class for mouse " + channel_number + " is " + model.states(predicted_class))
            data_points = data_points[1:]
            end_data_analysis = time.perf_counter()
            print("time doing file ops: " + str(end_data_analysis - start_data_analysis))

        epoch_count = epoch_count + 1
        end_time = time.perf_counter()
        print("time for iteration was: " + str(end_time - start_time))
        start_time = time.perf_counter()


lock = threading.Lock()
if Config.cycle_test_data:
    CycleTestData.cycle_test_files(lock)
cycle_files(lock)
