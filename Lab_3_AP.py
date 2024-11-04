"""
    Luca-Paul Florian,
    Anomaly Detection,
    BDTS, 1st Year
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score


def predict(classifier, X, X_test):
    classifier.fit(X)

    Y_pred = classifier.predict(X_test)
    Y_scores = classifier.decision_function(X_test)

    return Y_pred, Y_scores


# Exercise 2 - 2D Datapoints
def ex2_2d():
    n_samples = [500, 500]
    centers = [[10, 0], [0, 10]]
    std = 1

    X, _ = make_blobs(n_samples, centers=centers, cluster_std=std)

    c_rate = 0.02
    X_test = np.random.uniform(-10, 20, [1000, 2])

    _, axs = plt.subplots(1, 3, figsize=(17, 5))

    _, Y_scores = predict(IForest(contamination=c_rate), X, X_test)
    axs[0].scatter(X_test[:, 0], X_test[:, 1], c=Y_scores, cmap='viridis', s=10)
    axs[0].title.set_text("IForest")

    _, Y_scores = predict(DIF(contamination=c_rate), X, X_test)
    axs[1].scatter(X_test[:, 0], X_test[:, 1], c=Y_scores, cmap='viridis', s=10)
    axs[1].title.set_text("DIF")

    _, Y_scores = predict(LODA(contamination=c_rate), X, X_test)
    axs[2].scatter(X_test[:, 0], X_test[:, 1], c=Y_scores, cmap='viridis', s=10)
    axs[2].title.set_text("LODA")

    plt.show()


# Exercise 2 - 3D Datapoints
def ex2_3d():
    n_samples = [500, 500]
    centers = [[0, 10, 0], [10, 0, 10]]
    std = 1

    X, _ = make_blobs(n_samples, centers=centers, cluster_std=std)

    c_rate = 0.02
    X_test = np.random.uniform(-10, 20, [1000, 3])

    _, axs = plt.subplots(1, 3, figsize=(17, 5), subplot_kw=dict(projection="3d"))

    _, Y_scores = predict(IForest(contamination=c_rate), X, X_test)
    axs[0].scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=Y_scores, cmap='viridis', s=10)
    axs[0].title.set_text("IForest")

    _, Y_scores = predict(DIF(contamination=c_rate), X, X_test)
    axs[1].scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=Y_scores, cmap='viridis', s=10)
    axs[1].title.set_text("DIF")

    _, Y_scores = predict(LODA(contamination=c_rate), X, X_test)
    axs[2].scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=Y_scores, cmap='viridis', s=10)
    axs[2].title.set_text("LODA")

    plt.show()


def compute_stats(Y_true, Y_pred, Y_scores):
    tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    ba = (tpr + tnr) / 2
    auc = roc_auc_score(Y_true, Y_scores)

    return ba, auc


# Exercise 3
def ex3():
    data = loadmat("./shuttle.mat")
    X, Y = data["X"], data["y"]

    mean = np.mean(X)
    var = np.var(Y)
    X = (X - mean) / var

    test_size = int(0.4 * np.shape(X)[0])
    train_size = np.shape(X)[0] - test_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size)

    for i in range(10):
        print(f"--- Split {i} ---")

        classifier = IForest()
        classifier.fit(X_train)
        Y_pred = classifier.predict(X_test)
        Y_scores = classifier.decision_function(X_test)

        ba, auc = compute_stats(Y_test, Y_pred, Y_scores)
        print(f"IForest - BA: {ba}, AUC: {auc}")

        classifier = LODA()
        classifier.fit(X_train)
        Y_pred = classifier.predict(X_test)
        Y_scores = classifier.decision_function(X_test)

        ba, auc = compute_stats(Y_test, Y_pred, Y_scores)
        print(f"LODA - BA: {ba}, AUC: {auc}")

        classifier = DIF()
        classifier.fit(X_train)
        Y_pred = classifier.predict(X_test)
        Y_scores = classifier.decision_function(X_test)

        ba, auc = compute_stats(Y_test, Y_pred, Y_scores)
        print(f"DIF - BA: {ba}, AUC: {auc}")


if __name__ == "__main__":
    ex2_2d()
    ex2_3d()
    ex3()