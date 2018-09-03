from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline


def bad_usecase():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=41)

    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    svm = SVC()
    svm.fit(X_train_scaled, y_train)
    X_test_scaled = scaler.transform(X_test)

    print("Test score:{}".format(svm.score(X_test_scaled, y_test)))


def pipeline_usecase():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=41)

    pipe = create_pipeline()

    pipe.fit(X_train, y_train)
    print("Pipe score:{}".format(pipe.score(X_test, y_test)))


def create_pipeline():
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("svm", SVC())
    ])

    return pipe


def pipeline_grid_search():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=41)
    param_grid = {
        "svm__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "svm__gamma": [0.001, 0.01, 0.1, 1, 10, 100]
    }

    pipe = create_pipeline()
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best CV accuracy:{}".format(grid.best_score_))
    print("Test set score:{}".format(grid.score(X_test, y_test)))
    print("Best parameters:{}".format(grid.best_params_, ))


def pipeline_shorthand():
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=41)
    param_grid = {
        "svc__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "svc__gamma": [0.001, 0.01, 0.1, 1, 10, 100]
    }

    pipe = make_pipeline(MinMaxScaler(), SVC(C=100))
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Best CV accuracy:{}".format(grid.best_score_))
    print("Test set score:{}".format(grid.score(X_test, y_test)))
    print("Best parameters:{}".format(grid.best_params_, ))


#
# bad_usecase()
# pipeline_usecase()
# pipeline_grid_search()
pipeline_shorthand()
