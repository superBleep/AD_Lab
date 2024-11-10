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


# Exercise 1
def ex1():
    size = 500
    X, _ = make_blobs(size, centers=1)

    vectors = np.random.multivariate_normal([0, 0], np.identity(2), 5)
    projections = [] # Projections of points X for each vector
    for v in vectors:
        values = []

        for x in X:
            values.append(np.dot(x, v))

        projections.append(values)

    bins = 5
    buffer = 5
    probs_per_vector, edges_per_vector = [], []
    for p in projections:
        hist, bin_edges = np.histogram(p, bins, (np.min(p) - buffer, np.max(p) + buffer))

        probs_per_vector.append(hist / size)
        edges_per_vector.append(bin_edges)

    scores = []
    for i, x in enumerate(X):
        mean = 0

        for j in range(5):
            p = projections[j][i] # Projection of point i on vector j
            edges = edges_per_vector[j] # Histogram edges for vector j

            for k in range(len(edges) - 1):
                if p >= edges[k] and p < edges[k + 1]:
                    # Probability of bin k associated with the projection of point i
                    mean += probs_per_vector[j][k]

        scores.append(mean / 5) # Mean of probabilities for each histogram

    test_size = 500
    X_test = np.random.uniform(-3, 3, [test_size, 2])

    plt.figure(1)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap='viridis', s=10)
    plt.show()


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


def split_data(X, Y):
    test_size = int(0.4 * np.shape(X)[0])
    train_size = np.shape(X)[0] - test_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size)

    return X_train, X_test, Y_train, Y_test


def predict_compute(classifier, X_train, X_test, Y_test):
    classifier.fit(X_train)
    Y_pred = classifier.predict(X_test)
    Y_scores = classifier.decision_function(X_test)

    ba, auc = compute_stats(Y_test, Y_pred, Y_scores)

    return ba, auc


# Exercise 3
def ex3():
    data = loadmat("./shuttle.mat")
    X, Y = data["X"], data["y"]

    mean = np.mean(X)
    var = np.var(Y)
    X = (X - mean) / var

    X_train, X_test, _, Y_test = split_data(X, Y)

    ba, auc = predict_compute(IForest(), X_train, X_test, Y_test)
    print(f"IForest - BA: {ba}, AUC: {auc}")

    ba, auc = predict_compute(LODA(), X_train, X_test, Y_test)
    print(f"LODA - BA: {ba}, AUC: {auc}")

    ba, auc = predict_compute(DIF(), X_train, X_test, Y_test)
    print(f"DIF - BA: {ba}, AUC: {auc}")

    mean_ba_iforest, mean_ba_loda, mean_ba_dif = 0, 0, 0
    mean_auc_iforest, mean_auc_loda, mean_auc_dif = 0, 0, 0
    for i in range(10):
        X_train, X_test, _, Y_test = split_data(X, Y)

        ba, auc = predict_compute(IForest(), X_train, X_test, Y_test)
        mean_ba_iforest += ba
        mean_auc_iforest += auc

        ba, auc = predict_compute(LODA(), X_train, X_test, Y_test)
        mean_ba_loda += ba
        mean_auc_loda += auc

        ba, auc = predict_compute(DIF(), X_train, X_test, Y_test)
        mean_ba_dif += ba
        mean_auc_dif += auc

    print(f"IForest - Mean BA: {mean_ba_iforest / 10}, Mean AUC: {mean_auc_iforest / 10}")
    print(f"LODA - Mean BA: {mean_ba_loda / 10}, Mean AUC: {mean_auc_loda / 10}")
    print(f"DIF - Mean BA: {mean_ba_dif / 10}, Mean AUC: {mean_auc_dif}")


if __name__ == "__main__":
    ex1()
    ex2_2d()
    ex2_3d()
    ex3()