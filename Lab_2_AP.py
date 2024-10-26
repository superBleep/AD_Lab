"""
    Luca-Paul Florian,
    Anomaly Detection,
    BDTS, 1st Year
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from pyod.utils.data import generate_data_clusters
from pyod.models import knn, lof
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs


def leverage_scores(X):
    n = np.shape(X)[0]
    U, _, _ = np.linalg.svd(X)
    idx = np.diag_indices(n)

    return U[idx]


def generate_data_2d(x_var, eps_mean, eps_var):
    n = 100
    a, b = 2, 8

    x = np.random.normal(0, x_var, n)
    Eps = np.random.normal(eps_mean, eps_var, n)
    y = a * x + b + Eps

    return np.array((x, y)).T


def generate_data_3d(x1_var, x2_var, eps_mean, eps_var):
    n = 100
    a, b, c = 2, 8, 5

    x1 = np.random.normal(0, x1_var, n)
    x2 = np.random.normal(0, x2_var, n)
    Eps = np.random.normal(eps_mean, eps_var, n)
    y = a * x1 + b * x2 + c + Eps

    return np.array((x1, x2, y)).T


def plot_data(X, axs, pos, title, type):
    scores = leverage_scores(X)
    max_scores = 5 # Highest k leverage scores
    scatter_size = 9
    idx = np.argpartition(scores, -max_scores)[-max_scores:]
    colors = ["green"] * np.shape(X)[0]
    for i in idx:
        colors[i] = "red"

    if type == "2d":
        axs[pos].scatter(X[:, 0], X[:, 1], color=colors, s=scatter_size)
    elif type == "3d":
        axs[pos].scatter(X[:, 0], X[:, 1],  X[:, 2], color=colors, s=scatter_size)
    axs[pos].title.set_text(title)


def ex1():
    # Exercise 1

    # Linear model
    _, axs1 = plt.subplots(2, 2, figsize=(10, 8))

    # Regular data points
    X = generate_data_2d(1, 0, 1)
    plot_data(X, axs1, (0, 0), r"Regular ($\sigma^2_x = 1$, $\sigma^2_\epsilon = 1$)", "2d")

    # High variance on x
    X = generate_data_2d(5, 0, 1)
    plot_data(X, axs1, (0, 1), r"High x variance ($\sigma^2_x = 5$, $\sigma^2_\epsilon = 1$)", "2d")

    # High variance on y
    X = generate_data_2d(1, 0, 5)
    plot_data(X, axs1, (1, 0), r"High y variance ($\sigma^2_x = 1$, $\sigma^2_\epsilon = 5$)", "2d")

    # High variance on x and y
    X = generate_data_2d(5, 0, 5)
    plot_data(X, axs1, (1, 1), r"High x,y variance ($\sigma^2_x = 5$, $\sigma^2_\epsilon = 5$)", "2d")

    plt.show()

    # 2D model
    _, axs2 = plt.subplots(2, 2, figsize=(10, 8), subplot_kw=dict(projection="3d"))

    X = generate_data_3d(1, 1, 0, 1)
    plot_data(X, axs2, (0, 0), r"Regular ($\sigma^2_{x_1} = 1$, $\sigma^2_{x_2} = 1$, $\sigma^2_\epsilon = 1$)", "3d")

    X = generate_data_3d(5, 5, 0, 1)
    plot_data(X, axs2, (0, 1), r"High x1, x2 variance ($\sigma^2_{x_1} = 5$, $\sigma^2_{x_2} = 5$, $\sigma^2_\epsilon = 1$)", "3d")

    X = generate_data_3d(1, 1, 0, 5)
    plot_data(X, axs2, (1, 0), r"High y variance ($\sigma^2_{x_1} = 1$, $\sigma^2_{x_2} = 1$, $\sigma^2_\epsilon = 5$)", "3d")

    X = generate_data_3d(5, 5, 0, 5)
    plot_data(X, axs2, (1, 1), r"High x1,x2,y variance ($\sigma^2_{x_1} = 5$, $\sigma^2_{x_2} = 5$, $\sigma^2_\epsilon = 5$)", "3d")

    plt.show()


def predict_and_plot_ex2(X_train, X_test, Y_train, Y_test, n_neighbors):
    classifier = knn.KNN(n_neighbors=n_neighbors)
    classifier.fit(X_train)
    Y_train_pred, Y_test_pred = classifier.predict(X_train), classifier.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(Y_test, Y_test_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    ba = (tpr + tnr) / 2

    _, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    scatter_size = 7
    custom_legend = [ptch.Patch(color="green", label="Inlier"), ptch.Patch(color="red", label="Outlier")]

    colors = np.array(["green"] * np.shape(X_train)[0])
    colors[np.where(Y_train == 1)] = "red"
    axs[0, 0].scatter(X_train[:, 0], X_train[:, 1], c=colors, s=scatter_size)
    axs[0, 0].title.set_text("Ground truth labels - Training data")

    colors = np.array(["green"] * np.shape(X_train)[0])
    colors[np.where(Y_train_pred == 1)] = "red"
    axs[0, 1].scatter(X_train[:, 0], X_train[:, 1], c=colors, s=scatter_size)
    axs[0, 1].title.set_text("Predicted truth labels - Training data")
    
    colors = np.array(["green"] * np.shape(X_test)[0])
    colors[np.where(Y_test == 1)] = "red"
    axs[1, 0].scatter(X_test[:, 0], X_test[:, 1], c=colors, s=scatter_size)
    axs[1, 0].title.set_text("Ground truth labels - Test data")

    colors = np.array(["green"] * np.shape(X_test)[0])
    colors[np.where(Y_test_pred == 1)] = "red"
    axs[1, 1].scatter(X_test[:, 0], X_test[:, 1], c=colors, s=scatter_size)
    axs[1, 1].title.set_text("Predicted labels - Test data")

    plt.legend(handles=custom_legend)
    plt.suptitle(f"kNN outlier detection (k={n_neighbors}, ba={ba})")


def ex2():
    n_train = 400
    n_test = 200
    n_clusters = 2
    cont_rate = 0.1

    X_train, X_test, Y_train, Y_test = generate_data_clusters(n_train, n_test, n_clusters=n_clusters, contamination=cont_rate)

    predict_and_plot_ex2(X_train, X_test, Y_train, Y_test, 3)
    predict_and_plot_ex2(X_train, X_test, Y_train, Y_test, 4)
    predict_and_plot_ex2(X_train, X_test, Y_train, Y_test, 5)
    predict_and_plot_ex2(X_train, X_test, Y_train, Y_test, 6)

    plt.show()


def predict_and_plot_ex3(X, Y, cont_rate, k_knn, k_lof):
    classifier = knn.KNN(cont_rate)
    classifier.fit(X)
    Y_pred_knn = classifier.predict(X)

    classifier = lof.LOF(k_lof, contamination=cont_rate)
    classifier.fit(X)
    Y_pred_lof = classifier.predict(X)

    _, axs = plt.subplots(1, 2)
    
    scatter_size = 7
    custom_legend = [ptch.Patch(color="green", label="Inlier"), ptch.Patch(color="red", label="Outlier")]

    colors = np.array(["green"] * np.shape(X)[0])
    colors[np.where(Y_pred_knn == 1)] = "red"

    axs[0].scatter(X[:, 0], X[:, 1], c=colors, s=scatter_size)
    axs[0].title.set_text(f"kNN prediction (k={k_knn})")

    colors = np.array(["green"] * np.shape(X)[0])
    colors[np.where(Y_pred_lof == 1)] = "red"
    axs[1].scatter(X[:, 0], X[:, 1], c=colors, s=scatter_size)
    axs[1].title.set_text(f"LOF prediction (k={k_lof})")

    plt.legend(handles=custom_legend)


def ex3():
    n_samples = [200, 100]
    centers = [[-10, 10], [10, 10]]
    cluster_std = [2, 6]
    cont_rate = 0.07

    X, Y = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)

    predict_and_plot_ex3(X, Y, cont_rate, 5, 20)
    predict_and_plot_ex3(X, Y, cont_rate, 2, 40)

    plt.show()
    

def ex4():
    pass


if __name__ == '__main__':
    ex1()
    ex2()
    ex3()
    ex4()