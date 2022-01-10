import numpy as np
import pandas as pd

#combine the smoothed and the raw epoch
def expand_epochs(m):
    return pd.concat([m.Sxx_df.add_suffix('_sm'), m.multitaper_df], axis=1)