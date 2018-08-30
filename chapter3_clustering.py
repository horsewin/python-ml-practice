import matplotlib.pyplot as plt
import mglearn.datasets
import numpy as np
from sklearn.datasets import load_breast_cancer, make_moons, load_iris, make_blobs
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.metrics.cluster import adjusted_rand_score


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


def KMeansBadPractice2():
    X, y = make_moons(random_state=0, n_samples=200)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    kmeans.predict(X)

    mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_)


def AgglomerativeDemo():
    mglearn.plots.plot_agglomerative_algorithm()


def ScipyDendrogram():
    X, y = make_blobs(random_state=0, n_samples=12)
    linkage_array = ward(X)
    dendrogram(linkage_array)

    #
    ax = plt.gca()
    bounds = ax.get_xbound()
    ax.plot(bounds, [7.25, 7.25], '--', c='k')
    ax.plot(bounds, [1, 1], '--', c='k')


def DBSCANDemo():
    X, y = make_blobs(random_state=0, n_samples=12)
    dbscan = DBSCAN()
    dbscan.fit(X)
    clusters = dbscan.fit_predict(X)

    print("Cluster memberships:{}".format(clusters))

    X, y = make_moons(random_state=0, n_samples=400, noise=0.05)
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    clusters = dbscan.fit_predict(X_scaled)

    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)


def compareClusterScore():
    X, y = make_moons(random_state=0, noise=0.05, n_samples=200)
    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    fix, axes = plt.subplots(1, 4, figsize=(15, 3),
                             subplot_kw={'xticks': (), 'yticks': ()})

    algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(X))
    print("Random Clusters:{}".format(random_clusters))
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
    axes[0].set_title("Random assignment - API: {:.2f}".format(adjusted_rand_score(y, random_clusters)))

    for ax, algorithms in zip(axes[1:], algorithms):
        clusters = algorithms.fit_predict(X_scaled)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60)
        ax.set_title("{} - API: {:.2f}".format(algorithms.__class__.__name__,
                                               adjusted_rand_score(y, clusters)))


compareClusterScore()
plt.show()
