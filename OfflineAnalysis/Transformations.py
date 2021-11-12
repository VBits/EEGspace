from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
def train_lda(dataframe,labels,rand_idx,components=3):
    lda = LDA(n_components=components)
    return lda, lda.fit_transform(dataframe.iloc[rand_idx], labels[rand_idx])


def lda_transform_df(m,lda):
    # Create dataframe for LDs
    LD = lda.transform(m.Sxx_df)
    return pd.DataFrame(data=LD, columns=['LD1', 'LD2', 'LD3'], index=m.Sxx_df.index)
