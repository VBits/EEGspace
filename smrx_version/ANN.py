import Config
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
######################################
# classification with ANN
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import compute_class_weight
import numpy as np

from Common import train_lda, load_offline_data, plot_lda

ANNfolder = Config.base_path + '/ANN'

def try_ann(series, states, convert_states=True):
    states_numeric = np.array([s[0] for s in np.array(states)]) if convert_states else states
    rand_idx = np.random.choice(len(series), size=40000, replace=False)
    X = series.iloc[rand_idx]

    model = keras.Sequential([
        keras.layers.Input(shape=(len(series.T),)),
        keras.layers.Dense(32, activation='relu', name='1stHidden'),
        keras.layers.Dense(16, activation='relu', name='2ndHidden'),
        keras.layers.Dense(4)
    ])

    # Optional Plot model
    tf.keras.utils.plot_model(model, to_file=ANNfolder + 'model2.png', show_shapes=True)

    # compute weights for each class
    # state_codes = { 'SWS':0,
    #                  'REM':1,
    #                  'LMwake':2,
    #                  'HMwake':3
    #                  }

    y = states_numeric[rand_idx]

    state_numbers = np.unique(y)

    classWeight = compute_class_weight('balanced', state_numbers, y)
    # classWeight = dict(enumerate(classWeight))
    classWeight = dict(zip(state_numbers, classWeight))

    model.compile(optimizer='adam',
                  #             loss=tf.keras.losses.BinaryCrossentropy(
                  # from_logits=False, label_smoothing=0, reduction="auto", name="binary_crossentropy"),
                  #           loss='mae',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Load weights if you have trained the network previously
    # model.load_weights(ANNfolder + 'keras_model.h5')

    # Train the model
    model.fit(X.values, y,
              class_weight=classWeight,
              epochs=50)

    # test accuracy
    # test_loss, test_acc = model.evaluate(X.values.T,mh.state_df['clusters_knn'][rand_idx].values, verbose=2)
    test_loss, test_acc = model.evaluate(series, states_numeric, verbose=2)
    print('\nTest accuracy:', test_acc)

    # classify dataframe using ANN
    probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    predictions = probability_model.predict(series)
    state_predictions = np.argmax(predictions, axis=1)

    _, lda_encoded_data = train_lda(series, states_numeric)

    plot_lda(lda_encoded_data, state_predictions)

    # save model
    model.save(ANNfolder + 'keras_model_2.h5')

if __name__ == '__main__':
    multitaper, unsmoothed, smoothed, states = load_offline_data()
    try_ann(smoothed, states)
