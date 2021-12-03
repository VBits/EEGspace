import numpy as np

def get_random_idx(array, size=40000, Repeat=False):
    rand_idx = np.random.choice(array[100:-100].index, size, replace=Repeat)
    return rand_idx