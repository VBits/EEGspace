    def multitaper(self,window_length=None,window_step=None):
        if window_length is None:
            window_length = 4 * int(self.EEG_fs)
        if window_step is None:
            window_step = 2 * int(self.EEG_fs)
        window_starts = np.arange(0, len(self.EEG_data) - window_length + 1, window_step)

        EEG_segs = detrend(self.EEG_data[list(map(lambda x: np.arange(x, x + window_length), window_starts))])

        freqs, psd_est, var_or_nu = tsa.multi_taper_psd(EEG_segs, Fs=self.EEG_fs, NW=4, adaptive=False, jackknife=False,
                                                        low_bias=True)  # , dpss=dpss, eigvals=eigvals)

        # self.multitaper_df = pd.DataFrame(index=freqs, data=psd_est.T)
        time_idx = pd.date_range(start=self.start[0], freq='{}ms'.format(window_step/self.EEG_fs*1000), periods=len(psd_est))
        self.multitaper_df = pd.DataFrame(index=time_idx, data=psd_est,columns=freqs)

    def process_spectrum(self,):
        ## Normalize the data and plot density spectrogram
        def SG_filter(x):
            return scipy.signal.savgol_filter(x, 41, 2)

        # Log scale
        Sxx_df = 10 * np.log(self.multitaper_df.T)

        # horizontal axis (time)
        iterations = 4
        for i in range(iterations):
            Sxx_df = Sxx_df.apply(SG_filter, axis=1, result_type='expand')

        def density_calc(dataframe, boundary=(-100, 90)):
            # now calculate the bins for each frequency
            density_mat = []
            mean_density = []
            for i in range(len(dataframe.index)):
                density, bins = np.histogram(dataframe.iloc[i, :], bins=5000, range=boundary, density=True)
                density_mat.append(density)
                mean_density.append(dataframe.iloc[i, :].mean())
            density_mat = np.array(density_mat)
            bins = (bins[1:] + bins[:-1]) / 2
            return density_mat, bins

        density_mat, bins = density_calc(Sxx_df, boundary=(-100, 90))  # -1,1550

        density_df = pd.DataFrame(index=bins, data=density_mat.T, columns=self.multitaper_df.columns)
        for i in range(iterations):
            density_df = density_df.apply(SG_filter, axis=0, result_type='expand')

        baseline = np.argmax(density_df.values > 0.01, axis=0)

        norm = 0 - bins[baseline]
        Sxx_norm = Sxx_df.add(norm, axis=0)
        self.density_norm, self.power_bins = density_calc(Sxx_norm, boundary=(-25, 50))
        self.Sxx_norm = pd.DataFrame(data=Sxx_norm.T.values,columns=self.multitaper_df.columns,
                                     index=self.multitaper_df.index)


##############################
# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=3)

# Use previously assigned labels to fit LDA 
X_train = lda.fit_transform(mh.Sxx_norm.iloc[rand_idx], mh.state_df['4_states'][rand_idx])

# Use previously fitted LDA on a new mouse
LD = lda.transform(mh.Sxx_norm)
mh.LD_df = pd.DataFrame(data=LD, columns=['LD1', 'LD2','LD3'], index=mh.Sxx_norm.index)


################################
# ANN (not needed for the closed loop pipeline)
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import compute_class_weight


mh.state_df = pd.read_pickle(ANNfolder + 'states_{}_m{}.pkl'.format(Folder[:6],mh.pos))
mh.Sxx_norm = pd.read_pickle(ANNfolder + 'Sxx_norm_{}_m{}.pkl'.format(Folder[:6],mh.pos))


rand_idx = np.random.choice(len(mh.Sxx_norm), size=40000,replace=False)
X = mh.Sxx_norm.iloc[rand_idx]



model = keras.Sequential([
    keras.layers.Input(shape=(len(mh.Sxx_norm.T),)),
    keras.layers.Dense(32, activation='relu',name='1stHidden'),
    keras.layers.Dense(16, activation='relu',name='2ndHidden'),
    keras.layers.Dense(4)
])

# Plot model
tf.keras.utils.plot_model(model, to_file=ANNfolder+'model.png', show_shapes=True)


#compute weights for each class
state_codes = { 'SWS':0,
                 'REM':1,
                 'LMwake':2,
                 'HMwake':3
                 }
classWeight = compute_class_weight('balanced', np.unique(mh.state_df['4_states'][rand_idx].values), mh.state_df['4_states'][rand_idx].values)
classWeight = dict(zip(np.unique(mh.state_df['4_states'][rand_idx].values),classWeight))

# convert dictionary to numbers
for i in ['SWS','REM','LMwake','HMwake']:
    print (i)
    classWeight[state_codes[i]] = classWeight.pop(i)

# convert states_df to consistent numbering
mh.state_df['state_codes'] = mh.state_df['4_states'].map(state_codes)


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(X.values,mh.state_df['state_codes'][rand_idx].values,
          class_weight=classWeight,
          epochs=5)


#test accuracy  on the same or another mouse
test_loss, test_acc = model.evaluate(mh.Sxx_norm.values,mh.state_df['state_codes'].values, verbose=2)
print('\nTest accuracy:', test_acc)

# make predictions
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(X.values.T)


# save model
model.save('D:/Ongoing_analysis/200604_B6J_NO_Misting/Mouse_3_201009/keras_model')
