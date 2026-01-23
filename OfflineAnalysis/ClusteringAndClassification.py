"""
v.2022 - Not finalized yet
"""
import OfflineAnalysis.ANN as ANN
from OfflineAnalysis.GeneralUtils import *
from OfflineAnalysis.ProcessSmrx import *
from OfflineAnalysis.Transformations import *
from OfflineAnalysis.PlottingUtils import *
from OfflineAnalysis.NearestNeighbors import *
from Pipeline import DPA
import joblib


# import numpy as np #for grid search
# import os #for grid search
# from sklearn.metrics import silhouette_score #for grid search

######################################
#1. Get data for indicated genotype and channel.
# Preprocess data, or specify to load preprocessed data
m = get_mouse('GFAP',16,load=False)

#Create an extended dataframe that contains the smoothed and raw epochs
m.Sxx_ext = expand_epochs(m)
rand_idx = get_random_idx(m.Sxx_ext,size=20000)

# Optional: If need to cut out data after a certain time (ex: mouse died partway through)
# cutoff_time = "2025-02-13 12:00:00" #Time after which data is cut out
# m.Sxx_ext = m.Sxx_ext.loc[m.Sxx_ext.index <= cutoff_time]
# rand_idx = get_random_idx(m.Sxx_ext,size=30000)

# Optional: for GCaMP imaging etc, extract ttl signal. On rig2, used channel 8
m_ttl = get_TTL_epochs_from_channel(ttl_channel_idx=8, strain='GCaMP8m', pos=15, load=False)


############################################################
# 2. Create ANN
# model = ANN.create_model(m.Sxx_ext)
#
# # a. Train ANN and save weights
# ANN.standardize_state_codes(m.state_df)
# ANN.plot_model(model, BaseDir + ExpDir)
# classWeight = ANN.calculate_weights(m,rand_idx)
# model = ANN.train_model(model,m.Sxx_ext,m.state_df,classWeight,rand_idx)
# model.save_weights(offline_data_path+ 'weights_Sxx_ext.h5')
#
# # b. Load previously determined ANN weights
# model.load_weights(offline_data_path + 'weights_Sxx_ext.h5')
#
# # Get labels from ANN
# m.state_df = pd.DataFrame(index=m.Sxx_ext.index)
# m.state_df['ann_labels'] = ANN.get_labels(model,m.Sxx_ext)

############################################################
# 3. Create the LDA space

# a. Create LDA using the ANN labels (works better with noise free data)
# lda, X_train = train_lda(m.Sxx_ext,m.state_df['ann_labels'],rand_idx,components=3)

# b. Load a previously created LDA (works better with noisy data)
# lda_filename = 'D:/SleepHomeostasis_D/240125_TRAP_Kir21_MPOam/ephys/lda_240125_240307_TRAP_m1.joblib'
# lda_filename = 'D:/SleepHomeostasis_D/240125_TRAP_Kir21_MPOam/ephys/lda_240125_240307_TRAP_m4.joblib'
# lda_filename = 'D:/SleepHomeostasis_D/240125_TRAP_Kir21_MPOam/ephys/lda_240125_240307_TRAP_m2.joblib'

lda_filename = offline_data_path +'lda_average.joblib'
lda = joblib.load(lda_filename)
# Reduce dimensionality of data and save in a new dataframe
m.LD_df = lda_transform_df(m.Sxx_ext,lda)
# Evaluate LDA space
plot_LDA(m, rand_idx, savefigure=False)
plt.savefig(m.figureFolder+ 'LDA_NoLabels' + m.figure_tail, dpi=dpi)

############################################################
# 4. Density peak clustering
# Find density peaks in low dimensional space, tweak Z (Z ~1.0, k_max default usually 201)
est = DPA.DensityPeakAdvanced(Z=1.6, k_max=201) #est = DPA.DensityPeakAdvanced(Z=0.6, k_max=81)
est.fit(m.LD_df.loc[rand_idx])
# Plot DPA clusters on LDA
plot_DPA_LDA(m, rand_idx, est, savefigure=False, view_angles=(90, -67))

# OPTIONAL merge spurious clusters into 4 labels, labels:merged_labels
label_dict = {0:0,
              1:1,
              2:3,
              3:3,
              4:4,
}
est.labels_ = np.vectorize(label_dict.get)(est.labels_)
plot_DPA_LDA(m, rand_idx, est, savefigure=False, remapped=True, view_angles=(90, -67))

# OPTIONAL Update LDA using the DPA clusters
# lda, X_train = train_lda_dpa_labels(m.Sxx_ext,est,rand_idx,components=3)
# m.LD_df = lda_transform_df(m.Sxx_ext,lda)

############################################################
# To run a grid search for Z and k_max values:
# Define the parameter grid
Z_values = [0.6, 0.8, 1.0, 1.0, 1.2]
k_max_values = [51, 101, 151, 201]

# Store results
silhouette_results = []

for Z in Z_values:
    for k_max in k_max_values:
        print(f"\nRunning DPA with Z={Z}, k_max={k_max}")

        # Fit clustering
        est = DPA.DensityPeakAdvanced(Z=Z, k_max=k_max)
        est.fit(m.LD_df.loc[rand_idx])

        # Compute silhouette score
        try:
            score = silhouette_score(m.LD_df.loc[rand_idx], est.labels_)
        except ValueError as e:
            print(f"Silhouette score error: {e}")
            score = np.nan  # fallback if only one cluster is found

        silhouette_results.append({'Z': Z, 'k_max': k_max, 'score': score})
        print(f"Silhouette score: {score:.4f}" if not np.isnan(score) else "Score unavailable")

        # Plot and save the result
        plot_DPA_LDA(m, rand_idx, est, savefigure=False, view_angles=(90, -67))

        # Format filename safely
        z_str = f"{Z:.3f}".replace('.', 'p')  # e.g., "0.8" -> "0p800"
        filename = f"LDA_DPA_Z{z_str}_k{k_max}.png"
        plt.savefig(os.path.join(m.figureFolder, filename), dpi=OfflineConfig.dpi)

# Optional: print summary
print("\n=== Grid Search Summary ===")
for res in silhouette_results:
    print(f"Z={res['Z']}, k_max={res['k_max']}, score={res['score']}")



############################################################
# 5. Propagate DPC labels
knn_clf = get_knn_clf(m,rand_idx,est,n_neighbors=201) #default n_neighbors=201
# propagate labels
m.knn_pred(knn_clf, m.Sxx_ext,state_averages_path)
# Plot and evaluate state assignment 3D
plot_LDA(m,rand_idx,m.state_df['states'],savefigure=False)
plt.savefig(m.figureFolder + 'LDA_DPC_labels_{}_{}'.format(ExpDir[:6], File[:6]) + m.figure_tail, dpi=dpi)


# If states get swapped here, can correct manually like this:
# swap_dict = {"REM": "HTwake", "HTwake":"REM", } # specify states to swap. example: swap_dict = {"REM": "REM", "REM": "HTwake"}
# m.state_df["states"] = m.state_df["states"].replace(swap_dict)
# plot_LDA(m,rand_idx,m.state_df['states'],savefigure=False)
# plt.savefig(m.figureFolder + 'LDA_DPC_labels_{}_{}'.format(ExpDir[:6], File[:6]) + m.figure_tail, dpi=dpi)


# Note: if multiple states need to be swapped, as can happen in Kir2.1:
# clusters_per_state = m.state_df.groupby("states")["clusters_knn"].unique()
# cluster_to_state = {
#     0: "HTwake",
#     2: "LTwake",
#     3: "SWS"
# }
# m.state_df["states"] = m.state_df["clusters_knn"].map(cluster_to_state)
# plot_LDA(m,rand_idx,m.state_df['states'],savefigure=False)
# plt.savefig(m.figureFolder + 'LDA_DPC_labels_{}_{}'.format(ExpDir[:6], File[:6]) + m.figure_tail, dpi=dpi)
# 5b. Note, at this point can exclude outliers

############################################################
# 6. Save files
### -------------------
# Save or Load State Dataframe
m.state_df.to_pickle(BaseDir + ExpDir + 'states_{}_{}_{}_m{}.pkl'.format(ExpDir[:6], File[:6], m.genotype, m.pos))

# # #Load previously saved Dataframe from experimental folder
# m.state_df = pd.read_pickle(BaseDir + ExpDir + 'states_{}_{}_{}_m{}.pkl'.format(ExpDir[:6], File[:6], m.genotype, m.pos))
# ### -------------------
# # Store or load LDA transformation
# lda_filename = BaseDir + ExpDir + 'lda_{}_{}_{}_m{}.joblib'.format(ExpDir[:6], File[:6], m.genotype, m.pos)
# Save file
# joblib.dump(lda, lda_filename)
# # # Recover previously saved file
# lda = joblib.load(lda_filename)
# ### -------------------
# # Load OR Save knn model
# knn_file = BaseDir + ExpDir + 'knn_{}_{}_{}_m{}.joblib'.format(ExpDir[:6], File[:6], m.genotype, m.pos)
# # Save file
# joblib.dump(knn_clf, knn_file)
# # Recover previously saved file
# knn_file = offline_data_path +'knn_average.joblib'
# knn_clf = joblib.load(knn_file)


############################################################
#-------------------------------------------------
#Optional. Confusion matrix and accuracy score
#-------------------------------------------------
# from sklearn.metrics import confusion_matrix
# m.state_df = pd.read_pickle(BaseDir + ExpDir + 'states_{}_{}_{}_m{}.pkl'.format(ExpDir[:6], File[:6], m.genotype, m.pos))
# m.state_df_ground_truth = pd.read_pickle(BaseDir + ExpDir + 'states-corr_{}_{}_{}_m{}.pkl'.format(ExpDir[:6], File[:6], m.genotype, m.pos))
# confusion_matrix(m.state_df_ground_truth['states'],m.state_df['states'],normalize='true',labels=["SWS", "REM", "HTwake","LTwake"])
# accuracy_score(m.state_df_ground_truth['states'],m.state_df['states'])