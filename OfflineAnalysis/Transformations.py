from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd


def train_lda(dataframe,labels,rand_idx,components=3):
    lda = LDA(n_components=components)
    return lda, lda.fit_transform(dataframe.loc[rand_idx], labels.loc[rand_idx])


def lda_transform_df(dataframe,lda):
    # Create dataframe for LDs
    LD = lda.transform(dataframe)
    if lda.n_components == 2:
        LD_df = pd.DataFrame(data=LD, columns=['LD1', 'LD2'], index=dataframe.index)
    elif lda.n_components == 3:
        LD_df = pd.DataFrame(data=LD, columns=['LD1', 'LD2', 'LD3'], index=dataframe.index)
    return LD_df
