from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def train_lda(dataframe,labels,rand_idx,components=3):
    lda = LDA(n_components=components)
    return lda, lda.fit_transform(dataframe.iloc[rand_idx], labels[rand_idx])
