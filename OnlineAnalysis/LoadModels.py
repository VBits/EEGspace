"""
Online analysis
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import pickle
from OnlineAnalysis import Config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import Storage

#Standard behavioral state numbers
standardized_states = {
    "REM": 0,
    "SWS": 1,
    "LTwake": 2,
    "HTwake": 3,
    "LMwake": 2,
    "HMwake": 3,
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
def load_training_data_states(mouse_num):
    file_data = Storage.load_from_file(Config.state_file_path.format(mouse_num=mouse_num))
    state_data = np.array(file_data)
    states = list(set([(x[0], x[1]) for x in state_data]))
    states = {k: v for k, v in states}
    return states, np.array([s[0] for s in state_data]), get_standardized_state_mappings(states)

#Load multitaper data from offline analysis
def load_training_data(mouse_num):
    return Storage.load_from_file(Config.combined_data_file_path.format(mouse_num=mouse_num))

#Load LDA model from offline analysis
def get_lda_model(mouse_num, training_data, training_data_states):
    lda = Storage.load_from_file(Config.lda_file_path.format(mouse_num=mouse_num))
    lda_encoded_data = lda.transform(training_data)
    return lda, lda_encoded_data

#Load KNN model from offline analysis
def get_classification_model(mouse_num, training_data, training_data_states):
    return Storage.load_from_file(Config.knn_file_path.format(mouse_num=mouse_num))

#Create a MouseModel object to hold all the data for that mouse
class MouseModel:
    def __init__(self, mouse_num):
        self.training_data = load_training_data(mouse_num)
        self.states, self.training_data_states, self.state_mappings = load_training_data_states(mouse_num)
        self.get_standard_state_name = get_standard_state_name
        self.lda, self.lda_encoded_data = get_lda_model(mouse_num, self.training_data, self.training_data_states)
        self.classifier = get_classification_model(mouse_num, self.lda_encoded_data, self.training_data_states)
