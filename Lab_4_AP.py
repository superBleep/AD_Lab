"""
    Luca-Paul Florian,
    Anomaly Detection,
    BDTS, 1st Year
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import OneClassSVM
from scipy.io import loadmat


def compute_ba_auc(Y_true, Y_pred, Y_scores):
    tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    ba = (tpr + tnr) / 2
    auc = roc_auc_score(Y_true, Y_scores)

    return ba, auc


def compute_ba(Y_true, Y_pred):
    tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    return (tpr + tnr) / 2


def ex1():
    X_train, X_test, Y_train, Y_test = generate_data(300, 200, 3, 0.15)

    def plot_data(X, Y, axs, loc, title):
        colors = np.array(["green"] * np.shape(X)[0])
        colors[np.where(Y == 1)] = "red"
        axs[loc].scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=7)
        axs[loc].title.set_text(title)

    def apply_model(model, title):
        classifier = model
        classifier.fit(X_train)

        Y_train_pred = classifier.predict(X_train)
        Y_test_pred = classifier.predict(X_test)
        Y_test_scores = classifier.decision_function(X_test)

        test_ba, test_auc = compute_ba_auc(Y_test, Y_test_pred, Y_test_scores)

        print(f"BA (Test data): {test_ba}")
        print(f"ROC AUC (Test data): {test_auc}\n")

        _, axs = plt.subplots(2, 2, subplot_kw=dict(projection="3d"), figsize=(10, 8))
        legend = [ptch.Patch(color="green", label="Inlier"), ptch.Patch(color="red", label="Outlier")]

        plot_data(X_train, Y_train, axs, (0, 0), "Ground Truth - Training Data")
        plot_data(X_test, Y_test, axs, (0, 1), "Ground Truth - Test Data")
        plot_data(X_train, Y_train_pred, axs, (1, 0), "Prediction - Training data")
        plot_data(X_test, Y_test_pred, axs, (1, 1), "Prediction - Test data")

        plt.legend(handles=legend)
        plt.suptitle(title)
    
    apply_model(OCSVM("linear", contamination=0.15), "OCSVM (Linear kernel, cont=0.15)")
    apply_model(OCSVM("rbf", contamination=0.25), "OCSVM (RBF Kernel, cont=0.25)")
    apply_model(DeepSVDD(3, verbose=False), "Deep SVDD")

    plt.show()


def ex2():
    data = loadmat("./cardio.mat")
    X, Y = data["X"], data["y"]

    test_size = int(0.4 * np.shape(X)[0])
    train_size = np.shape(X)[0] - test_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size)

    parameters = {
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "gamma": [None, "scale" ,"auto", 0.3, None],
        "nu": [0.2, 0.3, 0.5, 0.6, 0.7],
    }

    estimator = OneClassSVM(kernel=parameters["kernel"], gamma=parameters["gamma"], nu=parameters["nu"])

    classifier = GridSearchCV(estimator, parameters, scoring=balanced_accuracy_score)
    classifier.fit(X_train)
    Y_scores = classifier.decision_function(X_test)

    print(Y_scores)


def ex3():
    data = loadmat("./shuttle.mat")
    X, Y = data["X"], data["y"]

    outliers = np.size(Y[np.where(Y == 1)])
    total = np.size(Y)
    cont_rate = np.round(outliers / total, 2)

    mean = np.mean(X)
    var = np.var(Y)
    X = (X - mean) / var

    test_size = int(0.5 * np.shape(X)[0])
    train_size = np.shape(X)[0] - test_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size)

    def compute_metrics(model):
        classifier = model
        classifier.fit(X_train)
        Y_test_pred = classifier.predict(X_test)
        Y_test_scores = classifier.decision_function(X_test)

        print(f"BA: {balanced_accuracy_score(Y_test, Y_test_pred)}")
        print(f"AUC: {roc_auc_score(Y_test, Y_test_scores)}\n")

    compute_metrics(OCSVM("linear", contamination=cont_rate))
    compute_metrics(DeepSVDD(np.shape(X)[1], contamination=cont_rate))


if __name__ == '__main__':
    #ex1()
    #ex2()
    ex3()