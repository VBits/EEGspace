#TODO Generate a file to be used for models while avoiding overfitting
def combine_files(EphysDir,Folder,save=True):
    os.chdir(EphysDir+Folder)

    if save:
        print ('saving file with states for all the mice')
        Sxx_combined = pd.DataFrame()
        for counter, file in enumerate(natsorted(glob.glob("Sxx*.pkl"))):
            print(file)
            Sxx_file = pd.read_pickle(file)
            mouse_id = file.split("_")[-1][:-4]
            states_combined["{}_states".format(mouse_id)] = states_file['states']

        states_combined.to_pickle(EphysDir + Folder + 'All_mice_states_{}_{}_{}.pkl'.format(Folder[:6],File[:6],mh.genotype))
    else:
        print ('loading file with states for all the mice')
        states_combined = pd.read_pickle(EphysDir + Folder + 'All_mice_states_combined')
        Sxx_combined = pd.read_pickle(EphysDir + Folder + 'All_mice_Sxx_combined.pkl')
    return states_combined, Sxx_combined


EphysDir = 'D:/Project_Mouse/Ongoing_analysis/'
Folder = 'Avoid_overfitting/'
