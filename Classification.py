def knn_prediction(clf):
    # predict in 2D
    self.state_df = pd.DataFrame(index=self.df.index)
    self.state_df['clusters_knn'] = clf.predict(self.pC)

    Nclusters = len(self.state_df['clusters_knn'].unique())

    # Count state instances after finding which code has higher average T_D.
    # Descending order(REM, Wake, SWS)
    state_code = np.zeros(Nclusters)
    for i in range(Nclusters):
        state_code[i] = np.mean(self.df['T_D_band'][self.state_df['clusters_knn'] == i])

    if Nclusters == 3:
        sws_code = np.argsort(state_code)[0]
        wake_code = np.argsort(state_code)[1]
        rem_code = np.argsort(state_code)[2]

        conditions = [(np.in1d(self.state_df['clusters_knn'], wake_code)),
                      (np.in1d(self.state_df['clusters_knn'], sws_code)),
                      (np.in1d(self.state_df['clusters_knn'], rem_code))]
    elif Nclusters == 4:
        sws_code = np.argsort(state_code)[0]
        LAwake_code = np.argsort(state_code)[1]
        HAwake_code = np.argsort(state_code)[2]
        rem_code = np.argsort(state_code)[3]

        conditions = [(np.in1d(self.state_df['clusters_knn'], [LAwake_code, HAwake_code])),
                      (np.in1d(self.state_df['clusters_knn'], sws_code)),
                      (np.in1d(self.state_df['clusters_knn'], rem_code))]
    else:
        print('Number of clusters not recognized. Run DPC again')

    state_choices = ['Wake', 'SWS', 'REM']

    self.state_df['3_states'] = np.select(conditions, state_choices, default="ambiguous")