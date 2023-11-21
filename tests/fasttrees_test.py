'''
Testing for Fast-And-Frugal tree module (fasttrees)
'''


from sklearn import datasets, model_selection
import pytest

from fasttrees.fasttrees import FastFrugalTreeClassifier



X_iris, y_iris = datasets.load_iris(as_frame=True, return_X_y=True)
classification_dataset = [(X_iris, y_iris)]


@pytest.mark.parametrize("X,y", classification_dataset)
def test_classification(X, y):
    '''Test fast-and-frugal classifier on the iris data set.

    For the given random state on the iris data set, the score should be higher than 0.6 for the train
    and test set.
    '''
    X_iris_train, X_iris_test, y_iris_train, y_iris_test = model_selection.train_test_split(
        X, y, test_size=0.4, random_state=42)

    fftc = FastFrugalTreeClassifier()
    fftc.fit(X_iris_train, y_iris_train)

    assert fftc.score(X_iris_train, y_iris_train) > 0.6
    assert fftc.score(X_iris_test, y_iris_test) > 0.6
