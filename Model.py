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


states = {
    "REM": 0,
    "SWS": 1,
    "LMwake": 2,
    "HMwake": 3,
}


def load_multiple_mouse_states(file_indexes, random_indexes):
    mouse_files = ['states_200604_m1.pkl',
                   'states_200604_m2.pkl',
                   'states_200604_m3.pkl',
                   'states_200604_m4.pkl',
                   'states_200424_m5.pkl',
                   'states_200424_m6.pkl',
                   'states_200424_m7.pkl']

    state_data = None

    for i, file in enumerate([mouse_files[i] for i in file_indexes]):
        f = open(Config.training_data_path + file, 'rb')
        s = pickle.load(f)
        rand_idx = random_indexes[i]
        s = np.array(s)[rand_idx]
        if state_data is None:
            state_data = np.empty((0, s.shape[1]))
        state_data = np.concatenate((state_data, s))

    return np.array([states[s[1]] for s in state_data])


def load_data_with_states(file_indexes=None):
    if file_indexes is None:
        file_indexes = [0, 1, 2, 3, 4, 5, 6]

    mouse_files = ['Sxx_norm_200604_m1.pkl',
                   'Sxx_norm_200604_m2.pkl',
                   'Sxx_norm_200604_m3.pkl',
                   'Sxx_norm_200604_m4.pkl',
                   'Sxx_norm_200424_m5.pkl',
                   'Sxx_norm_200424_m6.pkl',
                   'Sxx_norm_200424_m7.pkl']

    mouse_data = None

    random_indexes = []

    for file in [mouse_files[i] for i in file_indexes]:
        f = open(Config.training_data_path + file, 'rb')
        m = pickle.load(f)
        rand_idx = np.random.choice(m.shape[0], size=40000, replace=False)
        random_indexes.append(rand_idx)
        m = np.array(m)[rand_idx]
        if mouse_data is None:
            mouse_data = np.empty((0, m.shape[1]))
        mouse_data = np.concatenate((mouse_data, m))

    return mouse_data, load_multiple_mouse_states(file_indexes, random_indexes)


def get_lda_model(training_data, training_data_states):
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


def get_classification_model(training_data, training_data_states):
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


def get_model_object():
    model_path = Config.lda_model_path
    if Config.recreate_model_file or not os.path.isfile(model_path):
        model = Model()
        f = open(Config.lda_model_path, 'wb')
        pickle.dump(model, f)
    else:
        f = open(Config.lda_model_path, 'rb')
        model = pickle.load(f)
    return model


class Model:
    def __init__(self):
        self.training_data, self.training_data_states = load_data_with_states()
        self.lda, self.lda_encoded_data = get_lda_model(self.training_data, self.training_data_states)
        self.classifier = get_classification_model(self.lda_encoded_data, self.training_data_states)
        self.states = {v: k for k, v in states.items()}
