"""
    Luca-Paul Florian,
    Anomaly Detection,
    BDTS, 1st Year
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch


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


if __name__ == '__main__':
    ex1()