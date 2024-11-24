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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat


# Compute BA and ROC AUC metrics
def compute_ba_auc(Y_true, Y_pred, Y_scores):
    tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)

    ba = (tpr + tnr) / 2
    auc = roc_auc_score(Y_true, Y_scores)

    return ba, auc


# Exercise 1
def ex1():
    X_train, X_test, Y_train, Y_test = generate_data(300, 200, 3, 0.15)

    # Plot a subplot for a specific dataset + anomaly classification
    def plot_data(X, Y, axs, loc, title):
        colors = np.array(["green"] * np.shape(X)[0])
        colors[np.where(Y == 1)] = "red"
        axs[loc].scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, s=7)
        axs[loc].title.set_text(title)

    # Apply a prediciton model and compute the BA and ROC AUC metrics
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


# Exercise 2
def ex2():
    data = loadmat("./cardio.mat")
    X, Y = data["X"], data["y"]

    test_size = int(0.4 * np.shape(X)[0]) # 40% of the dataset
    train_size = np.shape(X)[0] - test_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size)

    # Parameter grid
    parameters = {
        "model__kernel": ["linear", "rbf", "sigmoid"],
        "model__gamma": ["scale" ,"auto", 0.3],
        "model__nu": [0.1, 0.2, 0.3, 0.5],
    }

    # Perform standardization before applying model
    pipeline = Pipeline([
        ('standardization', StandardScaler()),
        ('model', OneClassSVM())
    ])

    # Convert pyod anomaly labels to sklearn labels
    Y_train = 1 - (Y_train * 2)
    Y_test = 1 - (Y_test * 2)

    # Apply grid search and predict labels
    classifier = GridSearchCV(pipeline, parameters, scoring="balanced_accuracy")
    classifier.fit(X_train, Y_train)

    print("Best parameters: ", classifier.best_params_)
    
    Y_test_pred = classifier.predict(X_test)
    print("BA (Test data): ", balanced_accuracy_score(Y_test, Y_test_pred))
    

# Exercise 3
def ex3():
    data = loadmat("./shuttle.mat")
    X, Y = data["X"], data["y"]

    # Compute contamination rate (for models)
    outliers = np.size(Y[np.where(Y == 1)])
    total = np.size(Y)
    cont_rate = np.round(outliers / total, 2)

    # Normalize data
    mean = np.mean(X)
    var = np.var(Y)
    X = (X - mean) / var

    test_size = int(0.5 * np.shape(X)[0]) # 50% of the dataset
    train_size = np.shape(X)[0] - test_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size)

    # Apply model and compute BA and ROC AUC metrics
    def compute_metrics(model):
        classifier = model
        classifier.fit(X_train)
        Y_test_pred = classifier.predict(X_test)
        Y_test_scores = classifier.decision_function(X_test)

        print(f"BA: {balanced_accuracy_score(Y_test, Y_test_pred)}")
        print(f"AUC: {roc_auc_score(Y_test, Y_test_scores)}\n")

    compute_metrics(OCSVM("linear", contamination=cont_rate))
    compute_metrics(DeepSVDD(np.shape(X)[1], contamination=cont_rate))
    compute_metrics(DeepSVDD(np.shape(X)[1], epochs=50, batch_size=16, contamination=cont_rate, output_activation="relu"))
    compute_metrics(DeepSVDD(np.shape(X)[1], epochs=75, batch_size=64, contamination=cont_rate, output_activation="selu"))


if __name__ == '__main__':
    ex1()
    ex2()
    ex3()