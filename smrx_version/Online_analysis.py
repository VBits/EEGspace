

def chunker(seq, size):
    return (seq.iloc[pos:pos + size] for pos in range(0, len(seq)-1))

def SG_filter(x):
    return scipy.signal.savgol_filter(x, window_size, 2,mode='mirror')

#Log10
Sxx_df_raw = 10 * np.log(mh.multitaper_df)

#inputs
chunk_len = 41
smooth_iter = 3
window_size = 41
#creating a buffer to store the processed data
buffer = []
for n,chunk in enumerate(chunker(Sxx_df_raw[:50000],chunk_len)):
    print (n,len(chunk))
    #mirror the data to minimize bias at the end
    Sxx_mirror = pd.concat([chunk[:-1],chunk.iloc[::-1]],axis=0)
    #Savgol the mirrored df
    for i in range(smooth_iter):
        Sxx_mirror = scipy.signal.savgol_filter(Sxx_mirror, window_size, 2,mode='mirror',axis=0)
    #Get the savgol smoothed instance (== -1) or previous 10 instances (== -2... -11)
    Sxx_savgol_current = Sxx_mirror[chunk_len - 1]
    buffer.append(Sxx_savgol_current)

len(buffer)
buffer = np.asarray(buffer)
# Create dataframe for LDs, (use previously determined LDA, I uploaded an example)
LD_test = lda.transform(buffer)
LD_df_test = pd.DataFrame(data=LD_test, columns=['LD1', 'LD2','LD3'])
rand_idx = np.random.choice(len(LD_df_test), size=40000,replace=False)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(LD_df_test['LD1'][rand_idx],LD_df_test['LD2'][rand_idx],LD_df_test['LD3'][rand_idx],c='k',alpha=0.1, s=5)
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
ax.set_zlabel('LD3')