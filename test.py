import numpy as np
from IPython.display import display
import pandas as pd
import mglearn
import mglearn.datasets
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

#
# iris_dataset = load_iris()
# print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
# iris_dataframe = pd.DataFrame (X_train, columns=iris_dataset.feature_names)
# grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

# # P.32
X_forge, y_forge = mglearn.datasets.make_forge()
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
#
# plt.legend(["Class 0", "Class 1"], loc=4)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# plt.show()

# # P.33
# cancer = load_breast_cancer()
# print("cancer.key(): \n{}".format(cancer.keys()))
# print("Shape of cancer data:\n{}".format(cancer['data'].shape))
#
# boston = load_boston()
# print("boston.key(): \n{}".format(boston.keys()))
# print("Shape of boston data:\n{}".format(boston['data'].shape))
# X, y = mglearn.datasets.load_extended_boston()
# print("X.Shape:\n{}".format(X.shape))
# # mglearn.plots.plot_knn_classification(n_neighbors=3)
#
# # P.38
# X_forge_train, X_forge_test, y_forge_train, y_forge_test = train_test_split(X_forge, y_forge, random_state=0)
# clf = KNeighborsClassifier(n_neighbors=5)
# clf.fit(X_forge_train, y_forge_train)
# print("Test set prediction:\n{}".format(clf.predict(X_forge_test)))
# print("Test set accuracy:\n{}".format(clf.score(X_forge_test, y_forge_test)))

# P.39
# # K-NN Classifier
# flg, axes = plt.subplots(1, 3, figsize=(10, 3))
#
# for n_neighbors, ax in zip([1, 3, 15], axes):
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_forge, y_forge)
#     mglearn.plots.plot_2d_separator(clf, X_forge, fill=True, eps=0.5, ax=ax, alpha=.4)
#     mglearn.discrete_scatter(X_forge[:, 0], X_forge[:, 1], y_forge, ax=ax)
#     ax.set_title("{} neighbors".format(n_neighbors))
#     ax.set_xlabel("feature 0")
#     ax.set_ylabel("feature 1")
# axes[0].legend(loc=3)

# P.43
# K-NN Regressor
X_wave, y_wave = mglearn.datasets.make_wave(n_samples=40)
X_wave_train, X_wave_test, y_wave_train, y_wave_test = train_test_split(X_wave, y_wave, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_wave_train, y_wave_train)
print(":\n{}".format(reg.score(X_wave_test, y_wave_test)))

flg, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 15], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_wave_train, y_wave_train)

    ax.plot(line, reg.predict(line))
    ax.plot(X_wave_train, y_wave_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_wave_test, y_wave_test, 'v', c=mglearn.cm2(1), markersize=8)

    ax.set_title("{} neighbors".format(n_neighbors))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Traget")
axes[0].legend(loc="best")

plt.show()
