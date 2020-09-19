import nitime.algorithms as tsa
from scipy.signal import detrend


# Set widnow, overlap and calculate multitaper
window_length = 4 * int(mh.EEG_fs)
window_step = 2 * int(mh.EEG_fs)

window_starts = np.arange(0, len(mh.EEG_data) - window_length + 1, window_step)
EEG_segs = detrend(mh.EEG_data[list(map(lambda x: np.arange(x, x + window_length), window_starts))])
freqs, psd_est, var_or_nu = tsa.multi_taper_psd(EEG_segs, Fs=mh.EEG_fs,NW=4, adaptive=False, jackknife=False, low_bias=True)  # , dpss=dpss, eigvals=eigvals)
multitaper_df = pd.DataFrame(index=freqs,data=psd_est.T)

### Optional Plot multitaper spectrogram
plt.figure()
# normalize = mpl.colors.Normalize(vmin=0.1,vmax=600.515) #0.115
normalize = mpl.colors.LogNorm(vmin=12,vmax=200.4) #0.04
# p = plt.pcolormesh(window_starts[:10000],freqs,multitaper_df[np.arange(10000)], cmap='plasma',norm=normalize)
plt.title('Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.show()
cbar = plt.colorbar(p, label='Power [dB]')



## Normalize the data and plot density spectrogram
def SG_filter(x):
    return scipy.signal.savgol_filter(x, 41, 2)
# Log scale
Sxx = 10 * np.log(multitaper_df.values)

Sxx_df = pd.DataFrame(data=Sxx, index=mh.f)

# #alternative smoothing using median filter
# Sxx_median = scipy.ndimage.median_filter(Sxx,size=5)
# Sxx_df = pd.DataFrame(data=Sxx_median,index=mh.f)

# horizontal axis (time)
iterations = 4
for i in range(iterations):
    Sxx_df = Sxx_df.apply(SG_filter, axis=1, result_type='expand')

def density_calc(dataframe,boundary=(-100,90)):
    # now calculate the bins for each frequency
    density_mat = []
    mean_density = []
    for i in range(len(dataframe.index)):
        density, bins = np.histogram(dataframe.iloc[i, :], bins=5000, range=boundary, density=True)

        density_mat.append(density)
        mean_density.append(dataframe.iloc[i,:].mean())

    density_mat = np.array(density_mat)
    bins = (bins[1:]+bins[:-1])/2
    return density_mat, bins

density_mat, bins = density_calc(Sxx_df,boundary=(-100,90)) #-1,1550

density_df = pd.DataFrame(index=bins,data= density_mat.T,columns=mh.f)
for i in range(iterations):
    density_df = density_df.apply(SG_filter,axis=0,result_type='expand')

baseline = np.argmax(density_df.values>0.01,axis=0)

norm = 0 - bins[baseline]
Sxx_df_norm = Sxx_df.add(norm,axis=0)
density_mat_norm, bins_norm = density_calc(Sxx_df_norm,boundary=(-25,50))

plt.figure()
normalize = mpl.colors.Normalize(vmin=0,vmax=0.115)
p = plt.pcolormesh(mh.f,bins_norm,density_mat_norm.T, cmap='plasma',norm=normalize)
plt.title('Probability Density Spectrogram')
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.show()
cbar = plt.colorbar(p, label='Density')
plt.xlim(20,50)
plt.ylim(-25,50)
plt.savefig(figureFolder + 'Probability Density Normalized Spectrogram 0-50Hz 41SG 4 iter multitaper 4secWindow NW4' + figure_tail, dpi=dpi)


###### Do and visualize a PCA
pca = PCA(n_components=3)
pC = pca.fit_transform(Sxx_df_norm.values.T)
pC_df = pd.DataFrame(data=pC, columns=['pC1', 'pC2','pC3'])


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(pC_df['pC1'][rand_idx],pC_df['pC2'][rand_idx], pC_df['pC3'][rand_idx], c='k',alpha=0.1, s=4)
ax.set_xlabel('pC1')
ax.set_ylabel('pC2')
ax.set_zlabel('pC3')
plt.savefig(figureFolder+'PCA 3PCs multitaper 41SG 4iter 4secWindow' + figure_tail, dpi=dpi)

