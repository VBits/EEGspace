
####################
####################
####################
####################
# Plot a spectrogram with Freq and time on the two axes, probability density distribution on the color axis
# First you need to smoothen the spectrogram on the time axis
def SG_filter(x):
    return scipy.signal.savgol_filter(x, 101, 2)
#Log scale
Sxx = 10*np.log(mh.Sxx)
Sxx_df = pd.DataFrame(data=Sxx,index=mh.f)
#horizontal axis (time)
iterations=2
for i in range(iterations):
    Sxx_df = Sxx_df.apply(SG_filter,axis=1,result_type='expand')

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

plt.figure()
normalize = mpl.colors.Normalize(vmin=0,vmax=0.115) #0.115
# normalize = mpl.colors.LogNorm(vmin=0.001,vmax=0.74) #0.04
p = plt.pcolormesh(mh.f,bins,density_mat.T, cmap='plasma',norm=normalize)
plt.title('Probability Density Spectrogram')
# plt.ylabel('Power [$\mu V^2$]')
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.show()
cbar = plt.colorbar(p, label='Density')
plt.xlim(20,50)
plt.ylim(-75,50)
plt.savefig(figureFolder + 'Probability Density Spectrogram 0-50Hz Log scale 71SG 1 iteration' + figure_tail, dpi=dpi)

###################################################
#TODO detect an increase in density, smoothen the data on the power axis
density_df = pd.DataFrame(index=bins,data= density_mat.T,columns=mh.f)
for i in range(iterations):
    density_df = density_df.apply(SG_filter,axis=0,result_type='expand')

baseline = np.argmax(density_df.values>0.01,axis=0)

plt.figure()
normalize = mpl.colors.Normalize(vmin=0,vmax=0.115)
# normalize = mpl.colors.LogNorm(vmin=0.001,vmax=0.04)
p = plt.pcolormesh(mh.f,bins,density_df.values, cmap='plasma',norm=normalize)
plt.plot(mh.f,bins[baseline],color='tab:green',linewidth=2)
plt.title('Probability Density Spectrogram')
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.show()
cbar = plt.colorbar(p, label='Density')
plt.savefig(figureFolder + 'Probability Density Spectrogram 0-50Hz Log scale Norm line' + figure_tail, dpi=dpi)



norm = 0 - bins[baseline]
Sxx_df_norm = Sxx_df.add(norm,axis=0)
density_mat_norm, bins_norm = density_calc(Sxx_df_norm,boundary=(-25,50))


plt.figure()
normalize = mpl.colors.Normalize(vmin=0,vmax=0.115)
# normalize = mpl.colors.LogNorm(vmin=0.001,vmax=0.04)
p = plt.pcolormesh(mh.f,bins_norm,density_mat_norm.T, cmap='plasma',norm=normalize)
plt.title('Probability Density Spectrogram')
plt.ylabel('Power [dB]')
plt.xlabel('Frequency [Hz]')
plt.show()
cbar = plt.colorbar(p, label='Density')
plt.xlim(20,50)
plt.ylim(-25,50)
plt.savefig(figureFolder + 'Probability Density Normalized Spectrogram 0-50Hz' + figure_tail, dpi=dpi)

#TODO plot density per state


for i in [0,1,2,3]:
    b = np.array([mh.state_df['clusters_knn'].values == 2])
    Sxx_state = Sxx_df_norm.iloc[:, b.flatten()]
    density_state, bins_state = density_calc(Sxx_state, boundary=(-25, 50))

    plt.figure()
    normalize = mpl.colors.Normalize(vmin=0, vmax=0.185)
    # normalize = mpl.colors.LogNorm(vmin=0.001,vmax=0.04)
    p = plt.pcolormesh(mh.f, bins_state, density_state.T, cmap='plasma', norm=normalize)
    plt.title('Probability Density Spectrogram')
    plt.ylabel('Power [dB]')
    plt.xlabel('Frequency [Hz]')
    plt.show()
    cbar = plt.colorbar(p, label='Density')
    plt.xlim(20, 50)
    plt.xlim(0, 20)
    plt.ylim(-25, 50)
    plt.savefig(figureFolder + 'Probability Density Normalized Spectrogram 0-20Hz state3 HAwake' + figure_tail, dpi=dpi)

