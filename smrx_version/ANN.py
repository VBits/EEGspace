######################################3
# classification with ANN
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import compute_class_weight


mh.state_df = pd.read_pickle(ANNfolder + 'states_{}_m{}.pkl'.format(Folder[:6],mh.pos))
mh.Sxx_norm = pd.read_pickle(ANNfolder + 'Sxx_norm_{}_m{}.pkl'.format(Folder[:6],mh.pos))


rand_idx = np.random.choice(len(mh.Sxx_norm), size=40000,replace=False)
X = mh.Sxx_norm.iloc[rand_idx]


# X = X.div(50.8)


model = keras.Sequential([
    keras.layers.Input(shape=(len(mh.Sxx_norm.T),)),
    keras.layers.Dense(32, activation='relu',name='1stHidden'),
    keras.layers.Dense(16, activation='relu',name='2ndHidden'),
    keras.layers.Dense(4)
])

#Optional Plot model
tf.keras.utils.plot_model(model, to_file=ANNfolder+'model2.png', show_shapes=True)


#compute weights for each class
state_codes = { 'SWS':0,
                 'REM':1,
                 'LMwake':2,
                 'HMwake':3
                 }
classWeight = compute_class_weight('balanced', np.unique(mh.state_df['4_states'][rand_idx].values), mh.state_df['4_states'][rand_idx].values)
# classWeight = dict(enumerate(classWeight))
classWeight = dict(zip(np.unique(mh.state_df['4_states'][rand_idx].values),classWeight))

# convert dictionary to numbers
for i in ['SWS','REM','LMwake','HMwake']:
    print (i)
    classWeight[state_codes[i]] = classWeight.pop(i)

# convert states_df to consistent numbering
mh.state_df['state_codes'] = mh.state_df['4_states'].map(state_codes)


model.compile(optimizer='adam',
    #             loss=tf.keras.losses.BinaryCrossentropy(
    # from_logits=False, label_smoothing=0, reduction="auto", name="binary_crossentropy"),
    #           loss='mae',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Load weights if you have trained the network previously
model.load_weights(ANNfolder+ 'keras_model.h5')

# Train the model
model.fit(X.values,mh.state_df['state_codes'][rand_idx].values,
          class_weight=classWeight,
          epochs=5)


#test accuracy
test_loss, test_acc = model.evaluate(X.values.T,mh.state_df['clusters_knn'][rand_idx].values, verbose=2)
test_loss, test_acc = model.evaluate(mh.Sxx_norm.values,mh.state_df['state_codes'].values, verbose=2)
print('\nTest accuracy:', test_acc)

# make predictions for sample
X = mh.Sxx_norm.iloc[rand_idx]
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(X.values)
state_predictions = np.argmax(predictions,axis=1)

# classify dataframe using ANN
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(mh.Sxx_norm.values)
state_predictions = np.argmax(predictions,axis=1)


#plot state predictions
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(mh.LD_df['LD1'][rand_idx],mh.LD_df['LD2'][rand_idx], mh.LD_df['LD3'][rand_idx], c=state_predictions[rand_idx],alpha=0.2, s=4,cmap='Accent')
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_zlabel('C3')
plt.title('LDA')
plt.savefig(figureFolder+'LDA ANN clusters' + figure_tail, dpi=dpi)

# save model
model.save(ANNfolder+ 'keras_model_2.h5')




