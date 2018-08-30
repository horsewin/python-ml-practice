import matplotlib.pyplot as plt
import mglearn.datasets
import numpy as np
from IPython.display import display
from sklearn.datasets import load_breast_cancer, make_moons, load_iris, make_blobs
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def KMeansSimple():
    n = 3
    X, y = make_blobs(random_state=1)
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(X)
    print("Cluster Memberships:\n{}".format(kmeans.labels_))
    print("predict:\n{}".format(kmeans.predict(X)))

    mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
    mglearn.discrete_scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        range(0, n),
        markers='^', markeredgewidth=4
    )


def KMeansBadPractice():
    X, y = make_blobs(random_state=170, n_samples=600)
    rng = np.random.RandomState(74)
    transformation = rng.normal(size=(2, 2))
    X = np.dot(X, transformation)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')


KMeansBadPractice()
plt.show()
