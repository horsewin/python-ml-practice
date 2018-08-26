import matplotlib.pyplot as plt
import mglearn.datasets
import numpy as np
from IPython.display import display
from sklearn.datasets import load_breast_cancer, make_moons, load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def mlp():
    display(mglearn.plots.plot_single_hidden_layer_graph())


def showNonLinearFunc():
    line = np.linspace(-3, 3, 100)
    plt.plot(line, np.tanh(line), label="tanh")
    plt.plot(line, np.maximum(line, 0), label="relu")
    plt.legend(loc="best")
    plt.xlabel("x")
    plt.ylabel("relu(x), tanh(x)")


def neuralNetworkFunc():
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    mlp = MLPClassifier(
        solver="lbfgs",
        activation="tanh",
        random_state=0,
        hidden_layer_sizes=[10, 10]
    )
    mlp.fit(X_train, y_train)
    mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")


def cancerNeuralNetworkFunc():
    cancer = load_breast_cancer()
    print("cancer:{}".format(cancer.data.max(axis=0)))
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    mlp = MLPClassifier(random_state=42)
    mlp.fit(X_train, y_train)
    print("Accuracy on training set:{}".format(mlp.score(X_train, y_train)))
    print("Accuracy on test set:{}".format(mlp.score(X_test, y_test)))

    mean_on_train = X_train.mean(axis=0)
    std_on_train = X_train.std(axis=0)

    X_train_scaled = (X_train - mean_on_train) / std_on_train
    X_test_scaled = (X_test - mean_on_train) / std_on_train

    mlp = MLPClassifier(max_iter=1000, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    print("Accuracy on training set:{}".format(mlp.score(X_train_scaled, y_train)))
    print("Accuracy on test set:{}".format(mlp.score(X_test_scaled, y_test)))

    mlp = MLPClassifier(max_iter=1000, random_state=42, alpha=1)
    mlp.fit(X_train_scaled, y_train)
    print("Accuracy on training set:{}".format(mlp.score(X_train_scaled, y_train)))
    print("Accuracy on test set:{}".format(mlp.score(X_test_scaled, y_test)))

    plt.figure(figsize=(20, 5))
    plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
    plt.yticks(range(30), cancer.feature_names)
    plt.xlabel("Columns in weight matrix")
    plt.ylabel("Input feature")
    plt.colorbar()


def classConfidence():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)
    gbrt.fit(X_train, y_train)

    print("Decision Func shape:{}".format(gbrt.decision_function(X_test).shape))
    print("Decision Func:{}".format(gbrt.decision_function(X_test)[:6, :]))

    print("Argmax of decision func:\n{}".format(np.argmax(gbrt.decision_function(X_test), axis=1)))
    print("Predictions:\n{}".format(gbrt.predict(X_test)))


classConfidence()
plt.show()
