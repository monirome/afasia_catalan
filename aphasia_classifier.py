# aphasia_classifier.py

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def extract_features(file_path):
    data, sample_rate = sf.read(file_path)
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    return mfccs_processed

def prepare_dataset(df):
    features = []
    for index, row in df.iterrows():
        features.append(extract_features(row["name_chunk_audio"]))
    return np.array(features), np.array(df['aphasia_category'].tolist())


def train_aphasia_classifier(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = lgb.LGBMClassifier()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)]) #early_stopping_rounds=10,  verbose=True
    return clf


def predict_aphasia(clf, X):
    return clf.predict(X)


def evaluate_aphasia_classifier(clf, X, y):
    y_pred = predict_aphasia(clf, X)
    return accuracy_score(y, y_pred)


data_path = '/home/u917/PROJECT/aphasia/data_concat/concat_dataset.csv'  # Ruta del archivo CSV
df = pd.read_csv(data_path)

X, y = prepare_dataset(df)

classifier = train_aphasia_classifier(X, y)

accuracy = evaluate_aphasia_classifier(classifier, X, y)
print(f'Accuracy: {accuracy}')

# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
#
# def train_aphasia_classifier(X, y):
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#     clf = lgb.LGBMClassifier()
#     clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
#     return clf
#
# def predict_aphasia(clf, X):
#     return clf.predict(X)
#
# def evaluate_aphasia_classifier(clf, X, y):
#     y_pred = predict_aphasia(clf, X)
#     return accuracy_score(y, y_pred)
