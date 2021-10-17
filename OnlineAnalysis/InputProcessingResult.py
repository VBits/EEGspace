"""
Online analysis
"""


class InputProcessingResult:

    def __init__(self, mouse_number, epoch_count, standardized_class_number, standardized_class_name,
                 original_class_number, transformed_data, lda_point, raw_data, time_point):
        self.mouse_number = mouse_number
        self.epoch_count = epoch_count
        self.standardized_class_number = standardized_class_number
        self.standardized_class_name = standardized_class_name
        self.original_class_number = original_class_number
        self.transformed_data = transformed_data
        self.lda_point = lda_point
        self.raw_data = raw_data
        self.time_point = time_point

