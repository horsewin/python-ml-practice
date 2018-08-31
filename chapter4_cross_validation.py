import numpy as np
from sklearn.model_selection import cross_val_score, KFold, ShuffleSplit
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


# P.250
iris = load_iris()


def cross_validation(iris, cv):
    logreg = LogisticRegression()
    scores = cross_val_score(logreg, iris.data, iris.target, cv=cv)
    # print("Cross-validation scores:{}".format(scores))
    print("Cross-validation scores(means):{}".format(scores.mean()))


kfold = KFold(n_splits=5, shuffle=True, random_state=0)
shuffle_sprit = ShuffleSplit(n_splits=10, test_size=.5, train_size=.5)
cross_validation(iris, kfold)
cross_validation(iris, shuffle_sprit)

