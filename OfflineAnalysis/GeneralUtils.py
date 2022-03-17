import numpy as np
import pandas as pd

def get_random_idx(array, size=40000, Repeat=False):
    rand_idx = np.random.choice(array[100:-100].index, size, replace=Repeat)
    return rand_idx

#combine the smoothed and the raw epochs
def expand_epochs(m):
    return pd.concat([m.Sxx_df.add_suffix('_sm'), m.multitaper_df], axis=1)