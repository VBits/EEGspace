from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import pickle
import Config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import h5py
import joblib
import scipy




def get_standardized_states(states):
    standardized_states = {
        "REM": 0,
        "SWS": 1,
        "LMwake": 2,
        "HMwake": 3,
    }
    #todo transform the states to a standard form


def load_training_data_states(mouse_num):
    f = open(Config.state_file_path.format(mouse_num=mouse_num), 'rb')
    state_data = np.array(pickle.load(f))
    states = list(set([(x[0], x[1]) for x in state_data]))
    states = {k: v for k, v in states}
    return states, np.array([s[0] for s in state_data])


def load_training_data(mouse_num):
    f = open(Config.multitaper_data_file_path.format(mouse_num=mouse_num), 'rb')
    return pickle.load(f)


def get_lda_model(mouse_num, training_data, training_data_states):
    if Config.recreate_lda:
        #todo reshape this if we are going to use it again
        lda = LDA(n_components=3)
        lda_encoded_data = lda.fit_transform(training_data, training_data_states)
        plot_transformation = False
        if plot_transformation:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(lda_encoded_data[:, 0], lda_encoded_data[:, 1], lda_encoded_data[:, 2], c=training_data_states, alpha=0.1, s=8)
            ax.set_xlabel('component 1')
            ax.set_ylabel('component 2')
            ax.set_zlabel('component 3')
            plt.show()
        return lda, lda_encoded_data

    f = open(Config.lda_file_path.format(mouse_num=mouse_num), 'rb')
    lda = joblib.load(f)
    lda_encoded_data = lda.transform(training_data)
    return lda, lda_encoded_data


def get_classification_model(mouse_num, training_data, training_data_states):
    if Config.recreate_knn:
        X_train, X_test, y_train, y_test = train_test_split(training_data, training_data_states, test_size=0.01)
        classifier = KNeighborsClassifier(n_neighbors=8)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        result = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(result)
        result1 = classification_report(y_test, y_pred)
        print("Classification Report:", )
        print(result1)
        result2 = accuracy_score(y_test, y_pred)
        print("Accuracy:", result2)

        return classifier

    f = open(Config.knn_file_path.format(mouse_num=mouse_num), 'rb')
    return joblib.load(f)


def get_norm(mouse_num):
    return np.load(Config.norm_file_path.format(mouse_num=mouse_num), mmap_mode='r')


class Model:
    def __init__(self, mouse_num):
        self.training_data = load_training_data(mouse_num)
        self.states, self.training_data_states = load_training_data_states(mouse_num)
        self.lda, self.lda_encoded_data = get_lda_model(mouse_num, self.training_data, self.training_data_states)
        self.classifier = get_classification_model(mouse_num, self.lda_encoded_data, self.training_data_states)
        self.norm = get_norm(mouse_num)


def get_model_object(mouse_num):
    model_path = Config.mouse_model_path.format(mouse_num=mouse_num)
    if Config.recreate_model_file or not os.path.isfile(model_path):
        model = Model(mouse_num)
        f = open(model_path, 'wb')
        pickle.dump(model, f)
    else:
        f = open(model_path, 'rb')
        model = pickle.load(f)
    return model
