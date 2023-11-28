'''
Testing for Fast-And-Frugal tree module (fasttrees)
'''

import pathlib

from sklearn import datasets, model_selection
from sklearn.utils.estimator_checks import check_estimator
import pytest

import pandas as pd

from fasttrees import FastFrugalTreeClassifier



X_iris, y_iris = datasets.load_iris(as_frame=True, return_X_y=True)
classification_dataset = [(X_iris, y_iris)]


@pytest.mark.parametrize("estimator", [FastFrugalTreeClassifier])
def test_estimator(estimator):
    '''Test fast-and-frugal classifiers' compliance with scikit-learns estimator interface.
    '''
    return check_estimator(estimator)


@pytest.mark.parametrize("X,y", classification_dataset)
def test_classification(X, y):
    '''Test fast-and-frugal classifier on the iris data set.

    For the given random state on the iris data set, the score should be higher than 0.6 for the
    train and test set.
    '''
    y = y.apply(lambda entry: entry in [0, 1]).astype(bool)

    X_iris_train, X_iris_test, y_iris_train, y_iris_test = model_selection.train_test_split(
        X, y, test_size=0.4, random_state=42)

    fftc = FastFrugalTreeClassifier()
    fftc.fit(X_iris_train, y_iris_train)

    assert hasattr(fftc, 'classes_')
    assert hasattr(fftc, 'X_')
    assert hasattr(fftc, 'y_')

    y_pred_iris_train = fftc.predict(X)
    assert y_pred_iris_train.shape == (X.shape[0],)

    assert fftc.score(X_iris_train, y_iris_train) > 0.6
    assert fftc.score(X_iris_test, y_iris_test) > 0.6


def test_classification_heart_disease():
    '''Test fast-and-frugal classifier on the heart disease data set.

    The purpose is to test whether we get the same results as with the R implementation
    '''
    df_expected = pd.DataFrame(
        data={'IF NO': ['↓', 'decide NO', 'decide NO'],
              'feature': ['thal', 'cp', 'ca'],
              'direction': ['in', 'in', 'in'],
              'threshold': ["('fd', 'rd')", "('a',)", "('1', '2', '3')"],
              'IF YES': ['decide YES', '↓', 'decide YES']
    })
    df_expected.index.name = 'idx'

    words_expected = "If thal in ('fd', 'rd'), decide YES\n"\
                     "If not cp in ('a',), decide NO\n"\
                     "If not ca in ('1', '2', '3'), decide NO\n"\
                     "If ca in ('1', '2', '3'), decide NO, otherwise, decide YES\n"


    df_train = pd.read_csv(
        pathlib.Path('./data/heartdisease_train.csv'),
        dtype={
            'diagnosis': 'bool', # target
            'age': 'int',
            'sex': 'bool',
            'cp': 'category',
            'trestbps': 'int',
            'chol': 'int',
            'fbs': 'bool',
            'restecg': 'category',
            'thalach': 'int',
            'exang': 'bool',
            'oldpeak': 'float',
            'slope': 'category',
            'ca': 'category',
            'thal': 'category'
    })

    X, y = df_train.drop('diagnosis', axis=1), df_train['diagnosis']

    fftc = FastFrugalTreeClassifier()
    fftc.fit(X, y)

    y_pred = fftc.predict(X)
    assert y_pred.shape == (X.shape[0],)

    pd.testing.assert_frame_equal(
        fftc.get_tree().astype({'threshold': str}),
        df_expected)

    assert fftc.in_words() == words_expected
