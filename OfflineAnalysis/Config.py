from Mouse import Mouse

mh = Mouse("B6J", 1)

EphysDir = 'D:/Project_Mouse/Ongoing_analysis/'

Folder = '210409_White_noise/Ephys/'
File = '210409_000.smrx'

ANNfolder = 'D:/Project_mouse/Ongoing_analysis/ANN_training/'

FilePath = EphysDir+Folder+File


# standard functions for plotting
plot_kwds = {'alpha': 0.25, 's': 20, 'linewidths': 0}
# Set figure resolution
dpi = 500