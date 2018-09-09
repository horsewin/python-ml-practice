import numpy as np
from sklearn.model_selection import cross_val_score, KFold, ShuffleSplit, train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import mglearn
import mglearn.datasets
import matplotlib.pyplot as plt

iris = load_iris()

def grid_search(iris):
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

    best_score = 0
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            svm = SVC(gamma=gamma, C=C)
            svm.fit(X_train, y_train)

            score = svm.score(X_test, y_test)

            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}

    print("Best score:{:.2f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))

# grid_search(iris)

def correct_grid_search(iris):
    X_trainval, X_test, y_trainval, y_test = train_test_split(iris.data, iris.target, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=42)

    print("Size of training set: {}\nSize of validation set:{}\nSize of test set:{}"
          .format(X_train.shape[0], X_val.shape[0], X_test.shape[0]))

    best_score = 0
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            svm = SVC(gamma=gamma, C=C)
            svm.fit(X_train, y_train)

            # score = svm.score(X_val, y_val)
            scores = cross_val_score(svm, X_trainval, y_trainval, cv=5)
            score = np.mean(scores)

            if score > best_score:
                best_score = score
                best_parameters = {'C': C, 'gamma': gamma}

    svm = SVC(**best_parameters)
    svm.fit(X_trainval, y_trainval)
    test_score = svm.score(X_test, y_test)

    print("Best score:{:.2f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))
    print("Test score:{:.2f}".format(test_score))


correct_grid_search(iris)