from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def train_lda(dataframe,labels,components=3):
    lda = LDA(n_components=components)
    return lda.fit_transform(dataframe.iloc[rand_idx], labels[rand_idx])
