import numpy as np
import mglearn.datasets
import matplotlib.pyplot as plt
import graphviz
from sklearn.datasets import load_iris, load_breast_cancer, load_boston, make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestClassifier

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

classify_random_forest()
plt.show()

