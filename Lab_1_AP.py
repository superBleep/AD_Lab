import numpy as np
from matplotlib import pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve


def pred_plot(X, Y):
    clasifier = KNN()
    clasifier.fit(X)

    Y_pred = clasifier.predict(X)
    Y_scores = clasifier.predict(X)

    tn, fp, fn, tp = confusion_matrix(Y, Y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    ba = (tpr + tnr) / 2

    print(f"Balanced accuracy = {ba}")

    fpr, tpr, _ = roc_curve(Y, Y_scores)

    return fpr, tpr


def ex1_2():
    # Exercise 1
    train_size = 400
    test_size = 100
    cont_rate = 0.1

    X_train, X_test, Y_train, Y_test = generate_data(train_size, test_size, contamination=cont_rate)
    X_train_out = np.array([X_train[i] for i in range(train_size) if Y_train[i] == 1])

    plt.figure(1)
    plt.title("Training Data")
    plt.scatter(X_train[:,0], X_train[:,1], zorder=2, s=6)
    plt.scatter(X_train_out[:, 0], X_train_out[:, 1], color='red', zorder=2, s=6)
    plt.xlabel("Feature X")
    plt.xlabel("Feature Y")
    plt.show()

    # Exercise 2
    fpr1, tpr1 = pred_plot(X_train, Y_train)
    fpr2, tpr2 = pred_plot(X_test, Y_test)

    plt.figure(2)
    plt.title("ROC Curves")
    plt.plot(fpr1, tpr1, color='purple', label="Train data")
    plt.plot(fpr2, tpr2, color='yellow', label="Test data")
    plt.legend()
    plt.show()


def ex3():
    X_train, _, Y_train, _ = generate_data(1000, n_test=0)
        
    clasifier = KNN()
    clasifier.fit(X_train)
    Y_pred = clasifier.predict(X_train)

    Y = X_train - np.mean(X_train)
    Sigma = confusion_matrix(Y, Y_pred)
    W = np.linalg.solve(Sigma, Y)


if __name__ == '__main__':
    ex1_2()
    ex3()