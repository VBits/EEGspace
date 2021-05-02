import numpy as np
import Config
import time
import Modelling
import Preprocessing
from Timing import Timer
import sys
import Storage
from InputProcessingResult import InputProcessingResult

#arguments include defaulted libraries that can be replaced for the purposes of mocking in tests
def run_loop(mouse_number, queue, storage=Storage, modelling=Modelling, config=Config):
    sys.stdout.flush()
    model = modelling.get_model_for_mouse(mouse_number)
    epoch_count = 0
    data_points = []
    time_points = []
    total_points = 0
    spike_output_file_path = config.channel_file_base_path.format(channel_number=(mouse_number - 1))#zero indexed
    iteration = 0
    timer = Timer("start_time", mouse_number, iteration)
    dropped_epochs = []
    while storage.no_file_exists_at_location(spike_output_file_path):
        continue

    #record the timestamp before the first run and run it every 2 seconds after that
    seconds_per_iteration = config.num_seconds_per_epoch

    # file has been found, wait for it to be read
    time.sleep(0.30)
    # remove the first file
    storage.remove_file_if_exists(spike_output_file_path)

    iteration_deadline = time.time() + seconds_per_iteration

    while True:
        if time.time() > iteration_deadline:
            storage.remove_file_if_exists(spike_output_file_path)
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
        time_point, number_of_points_read, data_read = storage.consume_spike_output_data_file(spike_output_file_path)
        time_points.append(time_point)
        total_points += number_of_points_read
        data_points += data_read
        print("total points for mouse " + str(mouse_number) + " is " + str(total_points))
        timer.print_duration_since("start_reading_file", "Time doing file reading")

        if epoch_count > config.iteration_buffer:
            timer.set_time_point("start_data_analysis")
            transformed_data = Preprocessing.transform_data(data_points, timer, model.norm)
            transformed_data = [np.array(transformed_data)[-1]]
            lda_point = model.lda.transform(transformed_data)
            predicted_class = model.classifier.predict(lda_point)
            original_class_number = predicted_class[0]
            standardized_class_number = model.state_mappings[original_class_number]
            standardized_class_name = model.get_standard_state_name(standardized_class_number)
            input_processing_result = InputProcessingResult(mouse_number, epoch_count, standardized_class_number,
                                                            standardized_class_name, original_class_number)
            queue.put(input_processing_result)
            data_points = data_points[config.eeg_fs * config.num_seconds_per_epoch:]
            if mouse_number in config.print_timer_info_for_mice:
                print(len(data_points))
                print(dropped_epochs)
            timer.print_duration_since("start_data_analysis", "Time doing data analysis")

        epoch_count = epoch_count + 1
        sys.stdout.flush()
