"""
    Luca-Paul Florian,
    Anomaly Detection,
    BDTS, 1st Year
"""
import numpy as np
from matplotlib import pyplot as plt
from pyod.utils.data import generate_data
from pyod.models import knn
from sklearn.metrics import confusion_matrix, roc_curve
import random


def compute_thresh(Z_scores, Y):
    max_ba, id_thresh = 0, 0
    for i in range(101):
        thresh = np.quantile(Z_scores, i / 100)
        Y_pred = np.array([1. if z >= thresh else 0. for z in Z_scores])

        tn, fp, fn, tp = confusion_matrix(Y, Y_pred).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        ba = (tpr + tnr) / 2

        if ba > max_ba:
            max_ba = ba
            id_thresh = thresh

    return max_ba, id_thresh


def compute_vals(X_train, X, Y, cont_rate):
    clasifier = knn.KNN(cont_rate)
    clasifier.fit(X_train)

    Y_pred = clasifier.predict(X)
    Y_scores = clasifier.decision_function(X)

    tn, fp, fn, tp = confusion_matrix(Y, Y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    ba = (tpr + tnr) / 2

    fpr, tpr, _ = roc_curve(Y, Y_scores)

    return ba, fpr, tpr


def ex1_2():
    # Exercise 1
    train_size = 400
    test_size = 100
    cont_rate = 0.1

    X_train, X_test, Y_train, Y_test = generate_data(train_size, test_size, 2, cont_rate)
    X_train_out = np.array([X_train[i] for i in range(train_size) if Y_train[i] == 1])

    plt.figure(1)
    plt.title(f"Training Data (cont_rate={cont_rate})")
    plt.scatter(X_train[:,0], X_train[:,1], zorder=2, s=6, label="Inliers")
    plt.scatter(X_train_out[:, 0], X_train_out[:, 1], color='red', zorder=2, s=6, label="Outliers")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.legend()

    # Exercise 2
    ba_train, fpr_train, tpr_train = compute_vals(X_train, X_train, Y_train, cont_rate)
    ba_test, fpr_test, tpr_test = compute_vals(X_train, X_test, Y_test, cont_rate)

    print(f"BA (train data): {ba_train}")
    print(f"BA (test data): {ba_test}")

    plt.figure(2)
    plt.title(f"ROC curves (cont_rate={cont_rate})")
    plt.plot(fpr_train, tpr_train, color='purple', label="Train data")
    plt.plot(fpr_test, tpr_test, color='red', label="Test data")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()

    plt.figure(3)
    plt.title("ROC curves for multiple cont. rates")

    for i in range(1, 6):
        X_tr, X_t, _, Y_t = generate_data(train_size, test_size, 2, i / 10)
        _, fpr, tpr = compute_vals(X_tr, X_t, Y_t, i / 10)
        color = np.random.choice(range(100), 3) / 100

        plt.plot(fpr, tpr, color=color, label=f"{i / 10}")

    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()

    plt.show()

# Exercise 3
def ex3():
    data_size = 1000
    cont_rate = 0.1

    X, Y = generate_data(data_size, n_features=1, contamination=cont_rate, train_only=True)
        
    mean = np.mean(X)
    var = np.var(X)

    Z_scores = np.abs(X - mean) / var
    max_ba, id_thresh = compute_thresh(Z_scores, Y)

    print(f"Ideal Z-score threshold: {id_thresh}")
    print(f"Balanced accuracy: {max_ba}")


# Exercise 4
def ex4():
    data_size = 1000
    cont_rate = 0.1
    mean = np.array([3, 4])
    Sigma = np.array([
        [2, 1], 
        [1, 2]
    ])
    Z_scores = []

    X = np.random.multivariate_normal(mean, Sigma, data_size)
    Y = np.zeros(data_size)

    # Shift cont_rate % of data by 2 on both axes
    cont_idx = np.array(random.sample(range(1000), int(data_size * cont_rate)))
    for i in cont_idx:
        X[i] = np.array([X[i, 0] + 3, X[i, 1] + 3])
        Y[i] = 1

    L = np.linalg.cholesky(Sigma)
    for x in X:
        d = x - mean
        z = np.linalg.solve(L, d.T).T
        Z_scores.append(np.sqrt(z @ z))

    Z_scores = np.array(Z_scores)
    max_ba, id_thresh = compute_thresh(Z_scores, Y)

    print(f"Ideal Z-score threshold: {id_thresh}")
    print(f"Balanced accuracy: {max_ba}")


if __name__ == '__main__':
    ex1_2()
    ex3()
    ex4()