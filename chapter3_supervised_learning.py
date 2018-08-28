import matplotlib.pyplot as plt
import mglearn.datasets
import numpy as np
from IPython.display import display
from sklearn.datasets import load_breast_cancer, make_moons, load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def preprocessing():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    svm = SVC(C=100)
    svm.fit(X_train, y_train)
    print("Test set Accuracy:{}".format(svm.score(X_test, y_test)))


def preprocessingWithScaler(scaler):
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(C=100)
    svm.fit(X_train_scaled, y_train)
    print("Test set Accuracy:{}".format(svm.score(X_test_scaled, y_test)))


preprocessing()
preprocessingWithScaler(MinMaxScaler())
preprocessingWithScaler(StandardScaler())
