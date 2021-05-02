import sched, time
import random
#import WhiteNoiseOutput
import Storage

time_window = 10

def randomize_stimulus_output(stimulus_input_queue, stimulus_output_queue, stimulus_type):

    last_stimulus_timestamp = None
    s = sched.scheduler(time.time, time.sleep)
    stimulus_timepoints = []

    #todo use this enter method in the main loop
    def random_action():
        nonlocal last_stimulus_timestamp
        nonlocal stimulus_input_queue
        nonlocal stimulus_output_queue
        nonlocal stimulus_timepoints
        nonlocal stimulus_type
        if not stimulus_input_queue.empty():
            stimulus_input_queue = stimulus_input_queue.get()
            seconds_since_last_stimulus = (last_stimulus_timestamp - time.time())
            within_time_window = last_stimulus_timestamp is not None and seconds_since_last_stimulus > time_window
            can_output_stimulus = random.randint(30) == 1 and within_time_window
            if can_output_stimulus: #and stimulus_input_queue.standardized_class_name == "SWS":
                output_action(stimulus_type)
                t = time.time()
                last_stimulus_timestamp = t
                stimulus_timepoints.append(t)
                Storage.dump_to_file("last-run-timepoints.pkl", stimulus_timepoints)

        s.enter(60, 1, random_action)

    s.enter(60, 1, random_action)
    s.run()

def output_action(stimulus_type):
    if stimulus_type == "WhiteNoise":
        print("white noise!!!")
        #WhiteNoiseOutput.white_noise()