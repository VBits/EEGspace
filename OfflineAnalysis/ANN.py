# Initial classification with ANN
import tensorflow as tf
from tensorflow import keras
#TODO uncomment
# from Config import *
import numpy as np
from sklearn.utils import compute_class_weight
import pandas as pd
from OfflineAnalysis.PlottingUtils import *

#
# #previously trained network
# weights_file = 'keras_model.h5'



def create_model(dataframe):
    model = keras.Sequential([
    keras.layers.Input(shape=(len(dataframe.T),)),
    keras.layers.Dense(8096, activation='relu', name='1Hidden'),
    keras.layers.Dense(2048, activation='relu', name='2Hidden'),
    keras.layers.Dense(512, activation='relu', name='3Hidden'),
    keras.layers.Dense(126, activation='relu', name='4Hidden'),
    keras.layers.Dense(32, activation='relu', name='5Hidden'),
    keras.layers.Dense(16, activation='relu', name='6Hidden'),
    keras.layers.Dense(4)
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

    # # convert dictionary to numbers
    # for i in ['SWS', 'REM', 'LTwake', 'HTwake']:
    #     print(i)
    #     classWeight[state_codes[i]] = classWeight.pop(i)
    return classWeight

def train_model(model,dataframe,states,classWeight,rand_idx):
    # Train the model
    model.fit(dataframe.iloc[rand_idx].values, states['state_codes'].iloc[rand_idx].values,
                   class_weight=classWeight,
                   epochs=5)
    return model

def get_labels(model,m,Sxx_extended):
    # classify dataframe using ANN
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(Sxx_extended.values)
    m.state_df = pd.DataFrame(index=Sxx_extended.index)
    m.state_df['ann_states'] = np.argmax(predictions, axis=1)



def test_accuracy(model,m):
    test_loss, test_acc = model.evaluate(m.Sxx_df.values, m.state_df['state_codes'].values, verbose=2)
    print('\nTest accuracy:', test_acc)




