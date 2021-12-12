#Use SVM to get temporary labels
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
import joblib
import pandas as pd




def create_svm(dataframe,states,rand_idx,return_score=False,save=False):
    clf = make_pipeline(MinMaxScaler(), SVC(kernel='linear',probability=True,class_weight='balanced',random_state=42))
    # rand_idx = get_random_idx(dataframe,size=100000)
    X = dataframe.loc[rand_idx]
    y = states.loc[rand_idx]['states']
    clf.fit(X, y)
    if return_score:
        rand_idx = get_random_idx(dataframe, size=100000)
        X = dataframe.loc[rand_idx]
        y = states.loc[rand_idx]['states']
        print (clf.score(X,y))
    if save:
        svm_filename = offline_data_path + 'svm_all_mice.joblib'
        joblib.dump(clf, svm_filename)
    return clf

def load_svm(offline_data_path):
    svm_filename = offline_data_path + 'svm_all_mice.joblib'
    return joblib.load(svm_filename)

def get_svm_labels(array,clf):
    svm_labels = clf.predict(array)
    return pd.DataFrame(data=svm_labels,columns=['svm_labels'],index=array.index)

def get_svm_probabilities(array,clf):
    svm_probabilities = clf.predict_proba(array)
    return pd.DataFrame(data=svm_probabilities,columns=['svm_prob'],index=array.index)
    # return svm_probabilities