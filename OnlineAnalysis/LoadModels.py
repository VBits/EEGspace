"""
Online analysis
"""

from OnlineAnalysis import Config
import numpy as np

import Storage

#Standard behavioral state numbers
standardized_states = {
    "REM": 2,
    "SWS": 1,
    "LTwake": 0,
    "HTwake": 3,
}

#Remap state numbers
def get_standard_state_name(standardized_state_number):
    return [k for k, v in standardized_states.items() if v == standardized_state_number][0]

#Get standard state mappings
def get_standardized_state_mappings(states):
    state_mappings = {}
    for v, k in states.items():
        state_mappings[v] = standardized_states[k]
    return state_mappings

#Load states from offline analysis
def load_training_data_states(mouse_id):
    file_data = Storage.load_from_file(Config.state_file_path.format(mouse_id=mouse_id))
    state_data = np.array(file_data)
    states = list(set([(x[0], x[1]) for x in state_data]))
    states = {k: v for k, v in states}
    return states, np.array([s[0] for s in state_data]), get_standardized_state_mappings(states)

#Load multitaper data from offline analysis
def load_training_data(mouse_id):
    return Storage.load_from_file(Config.combined_data_file_path.format(mouse_id=mouse_id))

#Load LDA model from offline analysis
def get_lda_model(mouse_id, training_data, training_data_states):
    lda = Storage.load_from_file(Config.lda_file_path.format(mouse_id=mouse_id))
    lda_encoded_data = lda.transform(training_data)
    return lda, lda_encoded_data

#Load KNN model from offline analysis
def get_classification_model(mouse_id, training_data, training_data_states):
    return Storage.load_from_file(Config.knn_file_path.format(mouse_id=mouse_id))

#Create a MouseModel object to hold all the data for that mouse
class MouseModel:
    def __init__(self, mouse_id):
        self.training_data = load_training_data(mouse_id)
        self.states, self.training_data_states, self.state_mappings = load_training_data_states(mouse_id)
        self.get_standard_state_name = get_standard_state_name
        self.lda, self.lda_encoded_data = get_lda_model(mouse_id, self.training_data, self.training_data_states)
        self.classifier = get_classification_model(mouse_id, self.lda_encoded_data, self.training_data_states)
