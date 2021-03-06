"""
Online analysis
"""
import numpy as np
from OnlineAnalysis import Config, LoadModels, Preprocessing
import time
from OnlineAnalysis.Timing import Timer
import sys
import Storage
from OnlineAnalysis.InputProcessingResult import InputProcessingResult
from OnlineAnalysis.LoadModels import MouseModel


# arguments include defaulted libraries that can be replaced for the purposes of mocking in tests
def run_loop(mouse_number, queue, storage=Storage, load_models=LoadModels, config=Config):

    # set up variables
    model = MouseModel(mouse_number)

    data_points = []
    time_points = []
    dropped_epochs = []

    total_points = 0
    iteration = 1
    seconds_per_iteration = config.num_seconds_per_epoch
    spike_output_file_path = config.channel_file_base_path.format(channel_number=(mouse_number - 1))  # zero indexed

    timer = Timer("start_time", mouse_number, iteration)

    # remove any existing file
    storage.remove_file_if_exists(spike_output_file_path)

    # wait until a new file exists
    while storage.no_file_exists_at_location(spike_output_file_path):
        continue

    # file has been found, wait for it to be output fully
    time.sleep(0.50)

    # start the timer
    start_time = time.time()
    iteration_deadline = start_time + seconds_per_iteration

    while True:
        # if already past the deadline then record dropped epochs and reset the deadline
        if time.time() > iteration_deadline:
            num_epochs_dropped = int((time.time() - iteration_deadline) / seconds_per_iteration) + 1
            dropped_epochs_delay_seconds = num_epochs_dropped * seconds_per_iteration
            dropped_epochs = dropped_epochs + list(range(iteration, iteration + num_epochs_dropped))
            iteration += num_epochs_dropped - 1
            iteration_deadline += dropped_epochs_delay_seconds

        # wait until the next trigger time
        while time.time() < iteration_deadline:
            continue

        try:
            iteration_deadline += seconds_per_iteration
            timer.print_duration_since("start_time", "Time for iteration was")
            timer = Timer("start_time", mouse_number, iteration)

            timer.set_time_point("start_reading_file")
            time_point, number_of_points_read, data_read = storage.consume_spike2_output_data_file(
                spike_output_file_path)

            time_points.append(time_point)
            total_points += number_of_points_read
            data_points += data_read
            print("total points for mouse " + str(mouse_number) + " is " + str(total_points))
            timer.print_duration_since("start_reading_file", "Time doing file reading")

            if iteration > config.epoch_buffer:
                timer.set_time_point("start_data_analysis")
                transformed_data = Preprocessing.transform_data(data_points, timer, model.norm)
                transformed_data = [np.array(transformed_data)]
                lda_point = model.lda.transform(transformed_data)
                predicted_class = model.classifier.predict(lda_point)
                original_class_number = predicted_class[0]
                standardized_class_number = model.state_mappings[original_class_number]
                standardized_class_name = model.get_standard_state_name(standardized_class_number)
                input_processing_result = InputProcessingResult(mouse_number, iteration, standardized_class_number,
                                                                standardized_class_name, original_class_number,
                                                                transformed_data, lda_point, data_read, time_point)
                queue.put(input_processing_result)
                data_points = data_points[config.eeg_fs * config.num_seconds_per_epoch:]
                if mouse_number in config.print_timer_info_for_mice:
                    print(len(data_points))
                    print(dropped_epochs)
                timer.print_duration_since("start_data_analysis", "Time doing data analysis")

        except Exception as ex:
            dropped_epochs.append(iteration)
            #todo ensure the datapoint is the correct size here
            print("iteration encountered an error: ", str(iteration))
            print("Error details :", ex)
        finally:
            iteration += 1
            sys.stdout.flush()
