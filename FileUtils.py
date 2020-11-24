import joblib
import pickle
import numpy as np
import os


def load_or_recreate_file(path, recreate_function, recreate_file=False):
    if recreate_file or not os.path.isfile(path):
        object = recreate_function()
        dump_with_correct_lib(path, object)
    else:
        object = load_with_correct_lib(path)
    return object


def dump_with_correct_lib(path, object):
    f = open(path, 'wb')
    if path.endswith(".pkl"):
        pickle.dump(object, f)
    if path.endswith(".joblib"):
        joblib.dump(object, f)
    if path.endswith(".npy"):
        np.array(object).dump(path)


def load_with_correct_lib(path):
    f = open(path, 'rb')
    if path.endswith(".pkl"):
        return pickle.load(f)
    if path.endswith(".joblib"):
        return joblib.load(f)
    if path.endswith(".npy"):
        return np.load(f, allow_pickle=True)
