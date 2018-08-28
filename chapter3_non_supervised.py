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
from sklearn.decomposition import PCA


def samplePCA():
    mglearn.plots.plot_pca_illustration()
    plt.show()


def decomppose():
    # Scaling action
    cancer = load_breast_cancer()
    scaler = StandardScaler()
    scaler.fit(cancer.data)
    X_scaled = scaler.transform(cancer.data)

    # eliminate dimension
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    # the result of elimination
    print("Original shape:{}".format(str(X_scaled.shape)))
    print("Reduced shape:{}".format(X_pca.shape))

    # display figure
    plt.figure(figsize=(8, 8))
    mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
    plt.legend(cancer.target_names, loc="best")
    plt.gca().set_aspect("equal")
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")


decomppose()
plt.show()
