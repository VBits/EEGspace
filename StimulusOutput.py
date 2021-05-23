import sched, time
import random
import Storage
import sys
import WhiteNoiseOutput


def randomize_stimulus_output(stimulus_input_queue, stimulus_output_queue, stimulus_type):
    whitenoise_output = WhiteNoiseOutput.get_white_noise_function()

    def output_action():
        nonlocal stimulus_type
        nonlocal whitenoise_output
        if stimulus_type == "WhiteNoise":
            whitenoise_output()

    last_stimulus_timestamp = time.time()
    s = sched.scheduler(time.time, time.sleep)
    stimulus_timepoints = []

    # todo use this enter method in the main loop
    def random_action():
        time_window = 2  # 1 an hour
        try:
            nonlocal last_stimulus_timestamp
            nonlocal stimulus_input_queue
            nonlocal stimulus_output_queue
            nonlocal stimulus_timepoints
            nonlocal stimulus_type
            # if not stimulus_input_queue.empty():
            # stimulus_input_queue = stimulus_input_queue.get()
            seconds_since_last_stimulus = (time.time() - last_stimulus_timestamp)
            within_time_window = seconds_since_last_stimulus > time_window
            can_output_stimulus = random.randint(1, 3) == 1 and within_time_window
            if can_output_stimulus:  # and stimulus_input_queue.standardized_class_name == "SWS":
                output_action()
                t = time.time()
                last_stimulus_timestamp = t
                stimulus_timepoints.append(t)
                Storage.dump_to_file("last-run-timepoints.pkl", stimulus_timepoints)
                # save duration to filename or file entries if changing
        finally:
            sys.stdout.flush()

        s.enter(2, 1, random_action)

    s.enter(2, 1, random_action)
    s.run()
