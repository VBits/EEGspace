# Initial classification with ANN
import tensorflow as tf
from tensorflow import keras
from Config import ANNfolder
import numpy as np
from sklearn.utils import compute_class_weight

mh.state_df = pd.read_pickle(EphysDir+Folder + 'states_{}_m{}.pkl'.format(Folder[:6], mh.pos))
mh.Sxx_df = pd.read_pickle(EphysDir+Folder + 'Sxx_norm_{}_m{}.pkl'.format(Folder[:6], mh.pos))
weights_file = 'keras_model.h5'

# Save normalized Dataframe to experimental folder
mh.Sxx_norm.to_pickle(EphysDir + Folder + 'Sxx_norm_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], mh.genotype, mh.pos))
# Load previously saved Dataframe from experimental folder
mh.Sxx_norm = pd.read_pickle(EphysDir + Folder + 'Sxx_norm_{}_{}_{}_m{}.pkl'.format(Folder[:6], File[:6], mh.genotype, mh.pos))

class ANN:
    def __init__(self, dataframe, states):
        self.dataframe = dataframe
        self.states =states
    def initialize(self):
        self.model = keras.Sequential([
            keras.layers.Input(shape=(len(self.dataframe.T),)),
            keras.layers.Dense(32, activation='relu', name='1stHidden'),
            keras.layers.Dense(16, activation='relu', name='2ndHidden'),
            keras.layers.Dense(4)
        ])

        # Optional Plot model
        #tf.keras.utils.plot_model(model, to_file=ANNfolder + 'model2.png', show_shapes=True)


        self.model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])



    def train_model(self):
        # compute weights for each class
        state_codes = {'SWS': 0,
                       'REM': 1,
                       'LMwake': 2,
                       'HMwake': 3
                       }
        classWeight = compute_class_weight('balanced', np.unique(self.states[rand_idx].values),
                                           self.states[rand_idx].values)
        classWeight = dict(zip(np.unique(mh.state_df['4_states'][rand_idx].values), classWeight))

        # convert dictionary to numbers
        for i in ['SWS', 'REM', 'LMwake', 'HMwake']:
            print(i)
            classWeight[state_codes[i]] = classWeight.pop(i)

        # convert states_df to consistent numbering
        mh.state_df['state_codes'] = mh.state_df['4_states'].map(state_codes)





        # Load weights if you have trained the network previously
        model.load_weights(ANNfolder + 'keras_model.h5')

        # Train the model
        model.fit(X.values, mh.state_df['state_codes'][rand_idx].values,
                  class_weight=classWeight,
                  epochs=5)



    def get_labels(self,weights_file):
        #Load weights if you have trained the network previously
        self.model.load_weights(ANNfolder+ weights_file)

        # classify dataframe using ANN
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        predictions = probability_model.predict(dataframe.values)

        return np.argmax(predictions,axis=1)


    #TODO sandbox
        # test accuracy
        test_loss, test_acc = model.evaluate(X.values.T, mh.state_df['clusters_knn'][rand_idx].values, verbose=2)
        test_loss, test_acc = model.evaluate(mh.Sxx_norm.values, mh.state_df['state_codes'].values, verbose=2)
        print('\nTest accuracy:', test_acc)
