import io
import sys
import pandas as pd
import numpy as np
from Pipeline import DPA
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

m.LD_df.to_pickle(EphysDir + Folder + 'LD_df_{}_{}_{}_m{}.pkl'.format(Folder[:6],File[:6],m.genotype,m.pos))
# #Load previously saved Dataframe from experimental folder
n=1
# LD_df = pd.read_pickle(EphysDir + Folder + 'LD_df_211011_211102_SertCre-CS_m{}.pkl'.format(n))
Sxx_df = pd.read_pickle(EphysDir + Folder + 'Sxx_df_211011_211102_SertCre-CS_m{}.pkl'.format(n))
State_df = pd.read_pickle(EphysDir + Folder + 'states_211011_211102_SertCre-CS_m{}.pkl'.format(n))
# LD_df = pd.read_pickle(EphysDir + Folder + 'Multitaper_df_211011_211102_SertCre-CS_m1.pkl')
# LD_df_ds = LD_df.T.groupby(np.arange(len(LD_df.T))//2).mean()
# # LD_df_ds = LD_df_ds.T
x = MinMaxScaler().fit_transform(Sxx_df)
pca = PCA(n_components=4)
pC_df = pd.DataFrame(data=pca.fit_transform(x), index=LD_df.index)
# pC_df = pd.DataFrame(data=pca.fit_transform(Sxx_df), index=LD_df.index)
lda = LDA(n_components=3)
lda.fit(Sxx_df.loc[rand_idx], State_df['states'].loc[rand_idx])
LD_df = pd.DataFrame(data=lda.transform(Sxx_df), columns=['LD1', 'LD2', 'LD3'], index=Sxx_df.index)


# LD_df = pd.read_pickle(EphysDir + Folder + 'LD_df_211011_211102_SertCre-CS_m{}.pkl'.format(n))



# data_F1 = pd.read_csv("C:/DPA/src/Pipeline/tests/benchmarks/Fig1.dat", sep=" ", header=None)
# rand_idx = get_random_idx(Sxx_df,size=20000)
# plt.figure()
# plt.scatter(LD_df.loc[rand_idx].values[:,0],LD_df.loc[rand_idx].values[:,1],s=4,alpha=0.6,c='k')
# plt.figure()
# plt.scatter(data_F1.values[:,0],data_F1.values[:,1],s=4,alpha=0.6,c='k')



est = DPA.DensityPeakAdvanced(Z=0.17,k_max=201) #0.25

start=time.time()
# est.fit(data_F1)
est.fit(Sxx_df.loc[rand_idx])
# est.fit(pC_df.loc[rand_idx])
# est.fit(LD_df.loc[rand_idx])
end=time.time()
print(end-start)

# plt.figure()
# # plt.scatter(data_F1.values[:,0],data_F1.values[:,1],s=4,alpha=0.6,c=est.labels_,cmap='Accent')
# plt.scatter(LD_df.loc[rand_idx].values[:,0],LD_df.loc[rand_idx].values[:,1],s=4,alpha=0.6,c=est.labels_,cmap='Accent')

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(LD_df.loc[rand_idx].values[:,0], LD_df.loc[rand_idx].values[:,1], LD_df.loc[rand_idx].values[:,2],
           s=4,alpha=0.6,c=est.labels_,cmap='Accent')
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')

np.unique(est.halos_,return_counts=True)

est.topography_
# The topography can be visualized in the form of a dendrogram with the heights of the clusters proportional to the density of the centers.

plt.figure()
est.get_histogram()

est.

# Running again with a different Z without the need of recomputing the neighbors-densities

params = est.get_computed_params()
est.set_params(**params)
est.set_params(Z=1)
start=time.time()
est.fit(data_F1)
end=time.time()
print(end-start)

# The PAk and twoNN estimator can be used indipendently from the DPA clustering method.

# +
from Pipeline import PAk
from Pipeline import twoNN

rho_est = PAk.PointAdaptive_kNN()
d_est = twoNN.twoNearestNeighbors()


# +
results = rho_est.fit(data_F1)
print(results.densities_[:10])

dim = d_est.fit(data_F1).dim_
print(dim)
# -