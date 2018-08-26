import numpy as np
from IPython.display import display
import pandas as pd
import mglearn
import mglearn.datasets
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_wave()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)

lr = LinearRegression().fit(X_train, y_train)

print("Training set score:\n{}".format(lr.score(X_train,y_train)))
print("Test set score:\n{}".format(lr.score(X_test, y_test)))

X_boston, y_boston = mglearn.datasets.load_extended_boston()
X_boston_train, X_boston_test, y_boston_train, y_boston_test = train_test_split(X_boston, y_boston, random_state=0)
lr_boston = LinearRegression().fit(X_boston_train, y_boston_train)
ridge = Ridge().fit(X_boston_train, y_boston_train)
print("Training set score:\n{}".format(lr_boston.score(X_boston_train,y_boston_train)))
print("Test set score:\n{}".format(lr_boston.score(X_boston_test, y_boston_test)))

print("Training set score:{}".format(ridge.score(X_boston_train,y_boston_train)))
print("Test set score:{}".format(ridge.score(X_boston_test, y_boston_test)))

lasso = Lasso().fit(X_boston_train, y_boston_train)
print("Training set score:{}".format(lasso.score(X_boston_train,y_boston_train)))
print("Test set score:{}".format(lasso.score(X_boston_test, y_boston_test)))
print("The Number of features:{}".format(np.sum(lasso.coef_ != 0)))

lasso001 = Lasso(alpha=0.01, max_iter=1000000).fit(X_boston_train, y_boston_train)
print("Training set score:{}".format(lasso001.score(X_boston_train,y_boston_train)))
print("Test set score:{}".format(lasso001.score(X_boston_test, y_boston_test)))
print("The Number of features:{}".format(np.sum(lasso001.coef_ != 0)))

# P.57
X_forge, y_forge = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X_forge, y_forge)
    mglearn.plots.plot_2d_separator(clf, X_forge, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X_forge[:, 0], X_forge[:, 1], y_forge, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("First feature")
    ax.set_ylabel("Second feature")
axes[0].legend()

plt.show()