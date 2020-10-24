from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pickle
import Config
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Model:

    def __init__(self):
        self.lda, self.training_data = self.get_ldas_model()

    def load_multiple_mouse_states(self, file_indexes, random_indexes):
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

        return state_data

    def load_data_with_states(self, file_indexes=None):

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


    def get_ldas_model(self):
        model_path = Config.lda_model_path
        if not os.path.isfile(model_path):
            training_data, training_data_states = load_data_with_states()
            lda = LDA(n_components=3)
            X_train = lda.fit_transform(training_data, training_data_states)
            f = open(Config.lda_model_path, 'wb')
            pickle.dump(f)
            plot_transformation = True
            if plot_transformation:
                fig = plt.figure()
                ax = Axes3D(fig)
                colors = [d[c] for c in np.array(training_data_states[:, 1])]
                ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=colors, alpha=0.1, s=8)
                ax.set_xlabel('component 1')
                ax.set_ylabel('component 2')
                ax.set_zlabel('component 3')
                plt.show()
        else:
            f = open(Config.lda_model_path, 'rb')
            lda = pickle.load(f)
        return lda
