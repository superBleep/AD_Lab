import numpy as np
import matplotlib.pyplot as plt


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


def plot_data_2d(X, axs, pos, title):
    scores = leverage_scores(X)
    max_scores = 5
    scatter_size = 9
    xlim = [-5, 5]

    axs[pos].scatter(X[:, 0], X[:, 1], color='purple', s=scatter_size)
    idx = np.argpartition(scores, -max_scores)[-max_scores:]
    axs[pos].scatter(X[idx][:, 0], X[idx][:, 1], color='red', s=scatter_size)
    axs[pos].set_xlim(xlim)
    axs[pos].title.set_text(title)


def plot_data_3d(X, axs, pos, title):
    scores = leverage_scores(X)
    max_scores = 5
    scatter_size = 9
   
    axs[pos].projection = "3d"
    axs[pos].scatter(X[:, 0], X[:, 1],  X[:, 2], color='purple', s=scatter_size)
    idx = np.argpartition(scores, -max_scores)[-max_scores:]
    axs[pos].scatter(X[idx][:, 0], X[idx][:, 1],  X[idx][:, 2], color='red', s=scatter_size)
    axs[pos].title.set_text(title)


def ex1():
    _, axs1 = plt.subplots(2, 2, figsize=(10, 8))

    X = generate_data_2d(1, 0, 1)
    plot_data_2d(X, axs1, (0, 0), r"Regular ($\sigma^2_x = 1$, $\sigma^2_\epsilon = 1$)")

    X = generate_data_2d(5, 0, 1)
    plot_data_2d(X, axs1, (0, 1), r"High x variance ($\sigma^2_x = 5$, $\sigma^2_\epsilon = 1$)")

    X = generate_data_2d(1, 0, 5)
    plot_data_2d(X, axs1, (1, 0), r"High y variance ($\sigma^2_x = 1$, $\sigma^2_\epsilon = 5$)")

    X = generate_data_2d(5, 0, 5)
    plot_data_2d(X, axs1, (1, 1), r"High x,y variance ($\sigma^2_x = 5$, $\sigma^2_\epsilon = 5$)")

    _, axs2 = plt.subplots(2, 2, figsize=(10, 8))

    X = generate_data_3d(1, 1, 0, 1)
    plot_data_3d(X, axs2, (0, 0), r"Regular ($\sigma^2_x1 = 1$, $\sigma^2_x2 = 1$, $\sigma^2_\epsilon = 1$)")

    X = generate_data_3d(5, 5, 0, 1)
    plot_data_3d(X, axs2, (0, 1), r"High x1, x2 variance ($\sigma^2_x1 = 5$, $\sigma^2_x2 = 5$, $\sigma^2_\epsilon = 1$)")

    X = generate_data_3d(1, 1, 0, 5)
    plot_data_3d(X, axs2, (1, 0), r"High y variance ($\sigma^2_x1 = 1$, $\sigma^2_x2 = 1$, $\sigma^2_\epsilon = 5$)")

    X = generate_data_3d(5, 5, 0, 5)
    plot_data_3d(X, axs2, (1, 1), r"High x1,x2,y variance ($\sigma^2_x1 = 5$, $\sigma^2_x2 = 5$, $\sigma^2_\epsilon = 5$)")

    plt.show()


if __name__ == '__main__':
    ex1()