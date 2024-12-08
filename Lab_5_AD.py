"""
    Luca-Paul Florian,
    Anomaly Detection,
    BDTS, 1st Year
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
from scipy.io import loadmat
from pyod.utils.utility import standardizer
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras import Model, Sequential, layers


# Exercise 1
def ex1():
    # Step 1
    mean = [5, 10, 2]
    cov = [
        [3, 2, 2],
        [2, 10, 1],
        [2, 1, 2]
    ]
    size = 500

    X = np.random.multivariate_normal(mean, cov, size)

    fig = plt.figure(1, figsize=(15, 6))
    fig.subplots_adjust(left=0.02, right=0.974, wspace=0.338)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')

    ax1.scatter(X[:,0], X[:,1], X[:,2], c='blue', s=7)

    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    ax1.set_title('Generated dataset')
    
    X_c = X - np.mean(X) # Mean-center data
    Sigma = np.cov(X_c, rowvar=False) # Compute covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma) # EVD 

    # Step 2
    eigenvalues_desc = -np.sort(-eigenvalues)
    cum_exp_var = np.cumsum(eigenvalues_desc) / np.sum(eigenvalues_desc)
    ind_exp_var = eigenvalues_desc / np.sum(eigenvalues_desc)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.step(range(1, len(cum_exp_var) + 1), cum_exp_var)
    ax2.set_xticks(range(1, len(cum_exp_var) + 1))
    ax2.set_xlabel('Component')
    ax2.set_ylabel('Variance')
    ax2.set_title('Cummulative explained variances')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.bar(range(1, len(ind_exp_var) + 1), ind_exp_var,color='red')
    ax3.set_xticks(range(1, len(cum_exp_var) + 1))
    ax3.set_xlabel('Component')
    ax3.set_ylabel('Variance')
    ax3.set_title('Individual variances')

    plt.show()

    # Step 3
    X_proj = np.dot(X_c, eigenvectors[:, 2]) # Projection along 3rd component

    cont_rate = 0.1
    thresh = np.quantile(X_proj, cont_rate)

    fig2 = plt.figure(2, figsize=(12, 6))
    ax1 = fig2.add_subplot(1, 2, 1, projection='3d')

    idx = np.where(X_proj < thresh)[0]
    colors = ["green"] * np.shape(X)[0]
    for i in idx:
        colors[i] = "red"

    ax1.scatter(X[:,0], X[:,1], X[:,2], c=colors, s=7)

    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    ax1.set_title('PCA Anomaly Detection (3d principal component)')
    ax1.legend(handles=[ptch.Patch(color="green", label="Inlier"), ptch.Patch(color="red", label="Outlier")])

    X_proj = np.dot(X_c, eigenvectors[:, 1]) # Projection along 2nd component

    cont_rate = 0.1
    thresh = np.quantile(X_proj, cont_rate)

    ax2 = fig2.add_subplot(1, 2, 2, projection='3d')

    idx = np.where(X_proj < thresh)[0]
    colors = ["green"] * np.shape(X)[0]
    for i in idx:
        colors[i] = "red"

    ax2.scatter(X[:,0], X[:,1], X[:,2], c=colors, s=7)

    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_zlabel('Feature 3')
    ax2.set_title('PCA Anomaly Detection (2nd principal component)')
    ax2.legend(handles=[ptch.Patch(color="green", label="Inlier"), ptch.Patch(color="red", label="Outlier")])

    plt.show()

    # Step 4
    X_proj = np.dot(X_c, eigenvectors) # Project data along all components
    X_proj = np.dot(X_proj, np.sqrt(np.diag(eigenvalues))) # Normalize data

    centroid = np.mean(X_proj)
    dist = np.sum((X_proj - centroid) ** 2, axis=1)
    
    cont_rate = 0.1
    thresh = np.quantile(dist, cont_rate)

    fig3 = plt.figure(3)
    ax1 = fig3.add_subplot(1, 1, 1, projection='3d')

    idx = np.where(dist < thresh)[0]
    colors = ["green"] * np.shape(X)[0]
    for i in idx:
        colors[i] = "red"

    ax1.scatter(X[:,0], X[:,1], X[:,2], c=colors, s=7)

    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    ax1.set_title('PCA Anomaly Detection (Normalized distance)')
    ax1.legend(handles=[ptch.Patch(color="green", label="Inlier"), ptch.Patch(color="red", label="Outlier")])

    plt.show()


# Exercise 2
def ex2():
    # Step 1
    data = loadmat('./shuttle.mat')
    X, Y = np.array(data['X']), np.array(data['y'])
    N = np.shape(X)[0]

    X = standardizer(X) # Standardization

    # Compute real contamination rate
    outliers = np.size(Y[np.where(Y == 1)])
    total = np.size(Y)
    cont_rate = np.round(outliers / total, 2)

    test_size = int(0.6 * N) # 60% of data
    train_size = N - test_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size)

    pca = PCA(contamination=cont_rate)
    pca.fit(X_train)
    
    ind_exp_var = pca.explained_variance_
    cum_exp_var = np.cumsum(ind_exp_var) / np.sum(ind_exp_var)

    fig = plt.figure(1, figsize=(10, 5))

    ax2 = fig.add_subplot(1, 2, 1)
    ax2.step(range(1, len(cum_exp_var) + 1), cum_exp_var)
    ax2.set_xticks(range(1, len(cum_exp_var) + 1))
    ax2.set_xlabel('Component')
    ax2.set_ylabel('Variance')
    ax2.set_title('Cummulative explained variances')

    ax3 = fig.add_subplot(1, 2, 2)
    ax3.bar(range(1, len(ind_exp_var) + 1), ind_exp_var,color='red')
    ax3.set_xticks(range(1, len(cum_exp_var) + 1))
    ax3.set_xlabel('Component')
    ax3.set_ylabel('Variance')
    ax3.set_title('Individual variances')

    plt.show()

    # Step 2
    Y_train_pred, Y_test_pred = pca.predict(X_train), pca.predict(X_test)

    train_ba = balanced_accuracy_score(Y_train, Y_train_pred)
    test_ba = balanced_accuracy_score(Y_test, Y_test_pred)

    print("### BA (Train / Test sets, PCA) ###")
    print("Train: ", train_ba)
    print("Test: ", test_ba)

    kpca = KPCA(cont_rate, kernel='linear')
    kpca.fit(X_train)

    Y_train_pred, Y_test_pred = kpca.predict(X_train), kpca.predict(X_test)

    train_ba = balanced_accuracy_score(Y_train, Y_test_pred)
    test_ba = balanced_accuracy_score(Y_test, Y_test_pred)

    print("### BA (Train / Test sets, KPCA) ###")
    print("Train: ", train_ba)
    print("Test: ", test_ba)


# Step 2
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = Sequential([
            layers.Dense(8, 'relu'),
            layers.Dense(5, 'relu'),
            layers.Dense(3, 'relu')
        ])

        self.decoder = Sequential([
            layers.Dense(5, 'relu'),
            layers.Dense(8, 'relu'),
            layers.Dense(9, 'sigmoid')
        ])

    def call(self, inputs):
        encode = self.encoder(inputs)
        decode = self.decoder(encode)

        return decode


# Exercise 2
def ex3():
    # Step 1
    data = loadmat('./shuttle.mat')
    X, Y = np.array(data['X']), np.array(data['y'])
    N = np.shape(X)[0]

    X = MinMaxScaler((0, 1)).fit_transform(X) # Scale data to [0,1]

    test_size = int(0.6 * N) # 50% of data
    train_size = N - test_size
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, train_size=train_size)

    # Step 3
    autoencoder = Autoencoder()
    autoencoder.compile('adam', 'mse')
    history = autoencoder.fit(X_train, X_train, batch_size=1024, epochs=100, validation_data=(X_test, Y_test))

    plt.figure(1)

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')

    plt.xlabel('Epochs')
    plt.xlabel('Loss')
    plt.title('Training and validation losses')
    plt.legend()

    plt.show()

    # Step 4
    X_train_rec, X_test_rec = autoencoder.predict(X_train), autoencoder.predict(X_test)
    
    # Compute MSE
    train_rec_err = np.mean((X_train - X_train_rec) ** 2, axis=1)
    test_rec_err = np.mean((X_test - X_test_rec) ** 2, axis=1)

    # Compute contamination rate
    outliers = np.size(Y[np.where(Y == 1)])
    total = np.size(Y)
    cont_rate = np.round(outliers / total, 2)

    train_thresh = np.quantile(train_rec_err, cont_rate)
    test_thresh = np.quantile(test_rec_err, cont_rate)

    Y_train_pred = np.array(train_rec_err < train_thresh, dtype=int)
    Y_test_pred = np.array(test_rec_err < test_thresh, dtype=int)

    print("### BA (Train / Test set) ###")
    print("Train: ", balanced_accuracy_score(Y_train, Y_train_pred))
    print("Test: ", balanced_accuracy_score(Y_test, Y_test_pred))


if __name__ == '__main__':
    ex1()
    ex2()
    ex3()