import numpy as np
import mglearn.datasets
import matplotlib.pyplot as plt
import graphviz
from sklearn.datasets import load_iris, load_breast_cancer, load_boston, make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def DesicionTree():
    cancer = load_breast_cancer()
    X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = \
        train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

    # sample of over fitting
    # ex: pure tree
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_cancer_train, y_cancer_train)

    print("Accuracy on training set:{}".format(tree.score(X_cancer_train, y_cancer_train)))
    print("Accuracy on test set:{}".format(tree.score(X_cancer_test, y_cancer_test)))

    # max depth = 4
    tree = DecisionTreeClassifier(random_state=0, max_depth=4)
    tree.fit(X_cancer_train, y_cancer_train)

    print("Accuracy on training set:{}".format(tree.score(X_cancer_train, y_cancer_train)))
    print("Accuracy on test set:{}".format(tree.score(X_cancer_test, y_cancer_test)))
    #
    # export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
    #                 feature_names=cancer.feature_names, impurity=False, filled=True)
    print("Feature Importances:{}".format(tree.feature_importances_))

##
def plot_feature_importances_cancer(model):
    cancer = load_breast_cancer()
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")

def classify_random_forest():
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    forest = RandomForestClassifier(n_estimators=5, random_state=2)
    forest.fit(X_train, y_train)

    fix, axes = plt.subplots(2,3, figsize=(20, 10))
    for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
        ax.set_title("Tree {}".format(i))
        mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)

    mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1], alpha=.4)
    axes[-1, -1].set_title("Randam Forest")
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

# P.87 Gradient Boosting Tree
def GradientBoosting():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = \
        train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)
    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train, y_train)
    print("Accuracy on training set:{}".format(gbrt.score(X_train, y_train)))
    print("Accuracy on test set:{}".format(gbrt.score(X_test, y_test)))

    gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt.fit(X_train, y_train)
    print("Accuracy on training set:{}".format(gbrt.score(X_train, y_train)))
    print("Accuracy on test set:{}".format(gbrt.score(X_test, y_test)))


def SVM():
    X, y = mglearn.tools.make_handcrafted_dataset()
    svm = SVC(C=10, kernel='rbf', gamma=0.1).fit(X, y)
    mglearn.plots.plot_2d_separator(svm, X, eps=.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

    sv = svm.support_vectors_
    sv_labels = svm.dual_coef_.ravel() > 0
    mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")


def PlotSVMKernel():
    fix, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, C in zip(axes, [-1, 0, 3]):
        for a, gamma in zip(ax, range(-1, 2)):
            mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
    axes[0, 0].legend(["class0", "class1", "sv class 0", "sv class 1"], ncol=4, loc='best')


PlotSVMKernel()
plt.show()

