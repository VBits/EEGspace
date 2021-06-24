
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')
plt.rc('lines', linewidth=0.5)
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
import datetime
import os
from pydpc import Cluster
sys.path.append('C:/Users/bitsik0000/PycharmProjects/delta_analysis/SleepAnalysisPaper')
from functions import *
import matplotlib as mpl
import matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
### -------------------
# -----------------------------------------------------------------
# Inputs


mh = Mouse("B6J", 1)

#data directory
EphysDir = 'D:/Ongoing_analysis/'

Folder = '210409_White_noise/Ephys/'
File = '210409_000.smrx'

FilePath = EphysDir+Folder+File


#Create directory to save figures
figureFolder = mh.gen_folder(EphysDir,Folder)
mh.figure_tail = ' - {} - {}.png'.format(mh.pos, mh.genotype)
#Load data
mh.read_smrx(FilePath)
# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 500
# -----------------------------------------------------------------

###------------------------
#OPTIONAL remove drifting baseline
#This might be making 60Hz noise worse, so it needs to be called on demand
# _ , mh.EEG_data = iterative_savitzky_golay(mh,iterations=3)


#### -------------------
# # evaluate quality of recordings
matplotlib.use('Agg')
plt.figure()
plt.plot(mh.EEG_data)
plt.title('{}'.format(mh.Ch_name))
plt.ylabel(mh.Ch_units)
plt.ylim(1000,-1000)
plt.savefig(figureFolder+'{}_{}'.format(mh.Ch_name,File[:6]) + mh.figure_tail)
matplotlib.use('Qt5Agg')

### -------------------
### Downsample, perform multitaper and normalize data
target_fs=100
if mh.EEG_fs > target_fs:
    print ('downsampling mouse {} EEG data, from {}Hz to {}Hz'.format(mh.pos,mh.EEG_fs,target_fs))
    mh.downsample_EGG(target_fs=target_fs)

# resolution in seconds
mh.multitaper(resolution=2)
# specify amount of smoothing (depends also on resolution)
mh.process_spectrum(smooth_iter=4,window_size=41)

#Save normalized Dataframe to experimental folder
mh.Sxx_norm.to_pickle(EphysDir + Folder + 'Sxx_norm_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],mh.genotype,mh.pos))
#Load previously saved Dataframe from experimental folder
mh.Sxx_norm = pd.read_pickle(EphysDir + Folder + 'Sxx_norm_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],mh.genotype,mh.pos))

######################################
# 1. classification with ANN
import tensorflow as tf
from tensorflow import keras


model = keras.Sequential([
    keras.layers.Input(shape=(len(mh.Sxx_norm.T),)),
    keras.layers.Dense(32, activation='relu',name='1stHidden'),
    keras.layers.Dense(16, activation='relu',name='2ndHidden'),
    keras.layers.Dense(4)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

ANNfolder = 'D:/Ongoing_analysis/ANN_training/'
#Load weights if you have trained the network previously
model.load_weights(ANNfolder+ 'keras_model.h5')

# classify dataframe using ANN
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(mh.Sxx_norm.values)
state_predictions = np.argmax(predictions,axis=1)
#TODO
np.unique(state_predictions,return_counts=True)

######################################
# 2. Use ANN labels for LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

rand_idx = np.random.choice(len(mh.Sxx_norm), size=40000,replace=False)
lda = LDA(n_components=3)
X_train = lda.fit_transform(mh.Sxx_norm.iloc[rand_idx],state_predictions[rand_idx])

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_train[:,0],X_train[:,1], X_train[:,2], c='k',alpha=0.1, s=5)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
plt.title('LDA')
plt.savefig(figureFolder+'LDA after ANN labels' + mh.figure_tail, dpi=dpi)

# Create dataframe for LDs
LD = lda.transform(mh.Sxx_norm)
mh.LD_df = pd.DataFrame(data=LD, columns=['LD1', 'LD2','LD3'], index=mh.Sxx_norm.index)

######################################
# 3 Density peak clustering
# build the density peak clusterer
sample_data = np.ascontiguousarray(mh.LD_df.iloc[rand_idx].values)
clu = Cluster(sample_data)
plt.title('DPC', fontsize=15)
plt.savefig(figureFolder+'Density peaks' + mh.figure_tail)

# decide the cutoffs for the clusters
clu.assign(50,0.8)
plt.title('DPC boundaries', fontsize=15)
plt.savefig(figureFolder+'Density peak boundaries' + mh.figure_tail)

######################################
# 4. Propagate DPC labels
# Use Kneighbors to classify the rest of the data, use odd numbers to avoid draws
clf = KNeighborsClassifier(n_neighbors=201)
clf.fit(sample_data, clu.membership)

### -----
# predict states
mh.knn_pred(clf,transform='LDA')

### ----
# 3D
# Plot and evaluate state assignment 3D
fig = plt.figure()
ax = Axes3D(fig)
if 'states' in mh.state_df:
    ax.scatter(mh.LD_df['LD1'][rand_idx],mh.LD_df['LD2'][rand_idx], mh.LD_df['LD3'][rand_idx],
               c=mh.state_df['states'][rand_idx].apply(lambda x: mh.colors[x]),alpha=0.5, s=5,linewidths=0)
else:
    ax.scatter(mh.LD_df['LD1'][rand_idx], mh.LD_df['LD2'][rand_idx], mh.LD_df['LD3'][rand_idx],
               c=mh.state_df['clusters_knn'][rand_idx],cmap='tab10', alpha=0.5, s=5, linewidths=0)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')
plt.title('KNN')
plt.savefig(figureFolder+'LDA 3D states DPC labels with outliers {}_{}'.format(Folder[:6],File[:6]) + mh.figure_tail, dpi=dpi)


