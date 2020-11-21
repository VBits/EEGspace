# import h5py
# import joblib
# import pickle
#
# def load_or_recreate_file(path, recreate_function, recreate_file=False):
#     if recreate_file:
#         object = recreate_function()
#         f = open(model_path, 'wb')
#         pickle.dump(model, f)
#     else:
#         f = open(model_path, 'rb')
#         model = pickle.load(f)
#     return model
#
# def dump_with_correct_lib(path, object):
#
#
# def load_with_correct_object(path):
#