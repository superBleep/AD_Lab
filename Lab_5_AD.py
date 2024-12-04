"""
    Luca-Paul Florian,
    Anomaly Detection,
    BDTS, 1st Year
"""
import numpy as np
import matplotlib.pyplot as plt


# Exercise 1
def ex1():
    mean = [5, 10, 2]
    cov = [
        [3, 2, 2],
        [2, 10, 1],
        [2, 1, 2]
    ]
    size = 500

    X = np.random.multivariate_normal(mean, cov, size)

    fig = plt.figure(1)
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')

    ax1.scatter(X[:,0], X[:,1], X[:,2], c='green', s=7)

    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    
    X_c = X - np.mean(X)
    Sigma = np.cov(X_c)
    Delta, P = np.linalg.eigh(Sigma)

    Delta_desc = -np.sort(Delta)
    cum_exp_var = np.cumsum(Delta_desc) / np.sum(Delta_desc)
    ind_exp_var = Delta_desc / np.sum(Delta_desc)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.step(range(len(cum_exp_var)), cum_exp_var)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.bar(range(len(ind_exp_var)), ind_exp_var,color='red')

    plt.show()


if __name__ == '__main__':
    ex1()