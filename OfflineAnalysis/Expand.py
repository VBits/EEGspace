import numpy as np
import pandas as pd

# def chunker_list(seq, size):
#     return [seq[i:i + size] for i in range(len(seq) - 1)]
# def expand_epochs(m,smoothed_data=True,win=51):
#     if smoothed_data:
#         print ('Using savgol smoothed data')
#         data_chunks = chunker_list(m.Sxx_df,win)
#     else:
#         print ('Using multitaper data')
#         data_chunks = chunker_list(m.multitaper_df, win)
#     #trim off the epochs at the end that don't have the same length
#     data_chunks = data_chunks[:-win+1]
#     data_chunks = np.asarray(data_chunks)
#     #flatten the epochs into a single vector
#     extended_data = data_chunks.reshape(data_chunks.shape[0],-1)
#     #make epoch of interest the center one in the data
#     extended_data = pd.DataFrame(data=extended_data, index=m.Sxx_df[win//2:-win+win//2].index)
#     # state_data = m.state_df[win//2:-win+win//2]
#     return extended_data#, state_data

def expand_epochs(m):
    return pd.concat([m.Sxx_df.add_suffix('_sm'), m.multitaper_df], axis=1)