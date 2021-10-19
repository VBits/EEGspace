# Initial classification with ANN
import tensorflow as tf
from tensorflow import keras
from Config import ANNfolder
import numpy as np
from sklearn.utils import compute_class_weight
import pandas as pd
from OfflineAnalysis.PlottingUtils import PlottingUtils

mh.state_df = pd.read_pickle(EphysDir+Folder + 'states_210409_210409_B6Jv_m1.pkl')

#previously trained network
weights_file = 'keras_model.h5'

# Save normalized Dataframe to experimental folder
mh.Sxx_df.to_pickle(EphysDir + Folder + 'Sxx_df_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], mh.genotype, mh.pos))
# Load previously saved Dataframe from experimental folder
mh.Sxx_df = pd.read_pickle(EphysDir + Folder + 'Sxx_df_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], mh.genotype, mh.pos))

class ANN:
    def __init__(self, dataframe, state_df, rand_idx):
        self.dataframe = dataframe
        self.state_df = state_df
        self.rand_idx = rand_idx

    def initialize(self):
        self.model = keras.Sequential([
            keras.layers.Input(shape=(len(self.dataframe.T),)),
            keras.layers.Dense(2048, activation='relu', name='1aHidden'),
            keras.layers.Dense(1024, activation='relu', name='1bHidden'),
            keras.layers.Dense(254, activation='relu', name='1cHidden'),
            keras.layers.Dense(32, activation='relu', name='1stHidden'),
            keras.layers.Dense(16, activation='relu', name='2ndHidden'),
            keras.layers.Dense(4)
        ])


        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    def plot_model(self,Folder):
        tf.keras.utils.plot_model(self.model, to_file=Folder + 'ANN_model.png', show_shapes=True)

    def train_model(self):
        # compute weights for each class
        state_codes = {'SWS': 0,
                       'REM': 1,
                       'LTwake': 2,
                       'HTwake': 3
                       }
        classWeight = compute_class_weight('balanced', np.unique(self.state_df['states'][self.rand_idx].values),
                                           self.state_df['states'][self.rand_idx].values)
        classWeight = dict(zip(np.unique(self.state_df['states'][self.rand_idx].values), classWeight))

        # convert dictionary to numbers
        for i in ['SWS', 'REM', 'LTwake', 'HTwake']:
            print(i)
            classWeight[state_codes[i]] = classWeight.pop(i)

        # convert states to consistent numbering
        self.state_df['state_codes'] = self.state_df['states'].map(state_codes)

        # Train the model
        self.model.fit(self.dataframe.iloc[self.rand_idx].values, self.state_df['state_codes'][self.rand_idx].values,
                       class_weight=classWeight,
                       epochs=5)

    def get_labels(self):
        # classify dataframe using ANN
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(self.dataframe.values)

        self.state_df['ann_states'] = np.argmax(predictions, axis=1)

    def save_weights(self,Folder, weights_file):
        #Load weights if you have trained the network previously
        self.model.save_weights(Folder + weights_file)
    def load_weights(self,Folder, weights_file):
        #Load weights if you have trained the network previously
        self.model.load_weights(Folder+ weights_file)
    def test_accuracy(self):
        test_loss, test_acc = self.model.evaluate(self.dataframe.values, self.state_df['state_codes'].values, verbose=2)
        print('\nTest accuracy:', test_acc)




