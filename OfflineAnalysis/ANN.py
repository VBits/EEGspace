# Initial classification with ANN
import tensorflow as tf
from tensorflow import keras
#TODO uncomment
# from Config import *
import numpy as np
from sklearn.utils import compute_class_weight
import pandas as pd
from OfflineAnalysis.PlottingUtils import *


def create_model(dataframe):
    model = keras.Sequential([
    keras.layers.Input(shape=len(dataframe.T),name='Input'),
    keras.layers.Dense(126, activation='relu', name='1stHidden'),
    keras.layers.Dense(32, activation='relu', name='2ndHidden'),
    keras.layers.Dense(16, activation='relu', name='3rdHidden'),
    keras.layers.Dense(4,name='Output')
    ])


    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['SparseCategoricalAccuracy'])
    return model

def plot_model(model,Folder):
    tf.keras.utils.plot_model(model, to_file=Folder + 'ANN_model.png', show_shapes=True)

def standardize_state_codes(state_df):
    state_codes = {'SWS': 0,
                   'REM': 1,
                   'LTwake': 2,
                   'HTwake': 3
                   }
    # convert states to consistent numbering
    state_df['state_codes'] = state_df['states'].map(state_codes)

def calculate_weights(m,rand_idx):
    # compute weights for each class
    classWeight = compute_class_weight('balanced', np.unique(m.state_df['state_codes'][rand_idx].values),
                                       m.state_df['state_codes'][rand_idx].values)
    classWeight = dict(zip(np.unique(m.state_df['state_codes'][rand_idx].values), classWeight))

    return classWeight

def train_model(model,dataframe,states,classWeight,rand_idx,epochs=5):
    # Train the model
    model.fit(dataframe.loc[rand_idx].values, states['state_codes'].loc[rand_idx].values,
                   class_weight=classWeight,
                   epochs=epochs)
    return model

def get_labels(model,dataframe):
    # classify dataframe using ANN
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(dataframe.values)
    return np.argmax(predictions, axis=1)

def test_accuracy(model,m):
    test_loss, test_acc = model.evaluate(m.Sxx_ext.values, m.state_df['state_codes'].values, verbose=2)
    print('\nTest accuracy:', test_acc)




