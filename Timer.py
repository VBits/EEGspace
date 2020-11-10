import Config
import time


class Timer:

    def __init__(self, start_time_name, mouse_number, iteration_num):
        self.time_points = {start_time_name: time.perf_counter()}
        self.mouse_number = mouse_number
        self.iteration_num = iteration_num

    def set_time_point(self, time_point_name):
        self.time_points[time_point_name] = time.perf_counter()

    def print_duration_since(self, time_point_name, description=None):
        if self.mouse_number not in Config.print_timer_info_for_mice:
            return
        if description is None:
            description = "Time since " + time_point_name
        description = description + " for mouse " + \
                      str(self.mouse_number) + ", iteration " + str(self.iteration_num) + ": "
        print(description + str(time.perf_counter() - self.time_points[time_point_name]))
