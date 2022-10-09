# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 09:57:57 2022

@author: aleks
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def data_example_1(show_plot=True):
    # generating data
    np.random.seed(1)
    mu1 = np.array([0, 0])
    mu2 = np.array([20, 0])
    cov1 = np.array([[1, 0], [0, 1]])
    cov2 = np.array([[9, 0], [0, 9]])

    X1 = multivariate_normal.rvs(mu1, cov1, size=400, random_state=1)
    X2 = multivariate_normal.rvs(mu2, cov2, size=400, random_state=1)
    X = np.vstack((X1, X2))
    X = np.take(X, np.random.rand(X.shape[0]).argsort(), axis=0, out=X)

    if show_plot:
        plt.plot(X1[:, 0], X1[:, 1], '.', alpha=1, color="red", label=1)
        plt.plot(X2[:, 0], X2[:, 1], '.', alpha=1, color="blue", label=2)
        plt.title("True data")
        plt.legend()
        plt.grid()
        plt.show()
    return X


def data_example_1_3D(show_plot=True):
    # generating data
    np.random.seed(1)
    mu1 = np.array([0, 0, 0])
    mu2 = np.array([20, 0, 0])
    cov1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cov2 = np.array([[9, 0, 0], [0, 9, 0], [0, 0, 9]])

    X1 = multivariate_normal.rvs(mu1, cov1, size=400, random_state=1)
    X2 = multivariate_normal.rvs(mu2, cov2, size=400, random_state=1)
    X = np.vstack((X1, X2))
    X = np.take(X, np.random.rand(X.shape[0]).argsort(), axis=0, out=X)

    if show_plot:
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(X1[:, 0], X1[:, 1], X1[:, 2], c="red", cmap=plt.cm.Set1, edgecolor="k", s=40, label=1)
        ax.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c="blue", cmap=plt.cm.Set1, edgecolor="k", s=40, label=2)
        ax.legend()
        plt.title("True data")
        plt.grid()
        plt.show()
    return X


def data_example_2(show_plot=True):
    np.random.seed(1)
    mu1 = np.array([0, 3])
    mu2 = np.array([0, 5])
    mu3 = np.array([0, 7])
    mu4 = np.array([0, 5])
    cov = np.array([[1.2, 0], [0, 0.01]])
    cov2 = np.array([[0.01, 0], [0, 0.8]])

    X1 = multivariate_normal.rvs(mu1, cov, size=100, random_state=1)
    X2 = multivariate_normal.rvs(mu2, cov, size=100, random_state=1)
    X3 = multivariate_normal.rvs(mu3, cov, size=100, random_state=1)
    X4 = multivariate_normal.rvs(mu4, cov2, size=100, random_state=1)
    X = np.vstack((X1, X2, X3, X4))
    X = np.take(X, np.random.rand(X.shape[0]).argsort(), axis=0, out=X)
    if show_plot:
        plt.plot(X1[:, 0], X1[:, 1], '.', alpha=1, color="red", label=1)
        plt.plot(X2[:, 0], X2[:, 1], '.', alpha=1, color="blue", label=2)
        plt.plot(X3[:, 0], X3[:, 1], '.', alpha=1, color="green", label=3)
        plt.plot(X4[:, 0], X4[:, 1], '.', alpha=1, color="black", label=4)
        plt.title("True data")
        plt.grid()
        plt.legend()
        plt.show()
    return X


def data_example_3(show_plot=True):
    np.random.seed(1)
    mu1 = np.array([-4, -4])
    mu2 = np.array([-4, -4])
    mu3 = np.array([2, 2])
    mu4 = np.array([-1, -6])
    cov1 = np.array([[1, 0.5], [0.5, 1]])
    cov2 = np.array([[6, -2], [-2, 6]])
    cov3 = np.array([[2, -1], [-1, 2]])
    cov4 = np.array([[1 / 8, 0], [0, 1 / 8]])
    X1 = multivariate_normal.rvs(mu1, cov1, size=300, random_state=1)
    X2 = multivariate_normal.rvs(mu2, cov2, size=300, random_state=1)
    X3 = multivariate_normal.rvs(mu3, cov3, size=300, random_state=1)
    X4 = multivariate_normal.rvs(mu4, cov4, size=100, random_state=1)
    X = np.vstack((X1, X2, X3, X4))
    X = np.take(X, np.random.rand(X.shape[0]).argsort(), axis=0, out=X)
    if show_plot:
        plt.plot(X1[:, 0], X1[:, 1], '.', alpha=1, color="red", label=1)
        plt.plot(X2[:, 0], X2[:, 1], '.', alpha=1, color="blue", label=2)
        plt.plot(X3[:, 0], X3[:, 1], '.', alpha=1, color="green", label=3)
        plt.plot(X4[:, 0], X4[:, 1], '.', alpha=1, color="black", label=4)
        plt.title("True data")
        plt.grid()
        plt.legend()
        plt.show()
    return X


def data_example_5(show_plot=True):
    np.random.seed(1)
    mu1 = np.array([0, 0])
    mu2 = np.array([0, 0])
    mu3 = np.array([-1.5, 1.5])
    mu4 = np.array([1.5, 1.5])
    mu5 = np.array([0, -2])
    cov1 = np.array([[0.01, 0], [0, 1.25]])
    cov2 = np.array([[8, 0], [0, 8]])
    cov3 = np.array([[0.2, 0], [0, 0.015]])
    cov4 = np.array([[0.2, 0], [0, 0.015]])
    cov5 = np.array([[1, 0], [0, 0.2]])

    X1 = multivariate_normal.rvs(mu1, cov1, size=200, random_state=1)
    X2 = multivariate_normal.rvs(mu2, cov2, size=200, random_state=1)
    X3 = multivariate_normal.rvs(mu3, cov3, size=200, random_state=1)
    X4 = multivariate_normal.rvs(mu4, cov4, size=200, random_state=1)
    X5 = multivariate_normal.rvs(mu5, cov5, size=200, random_state=1)
    X = np.vstack((X1, X2, X3, X4, X5))
    X = np.take(X, np.random.rand(X.shape[0]).argsort(), axis=0, out=X)
    if show_plot:
        plt.plot(X1[:, 0], X1[:, 1], '.', alpha=1, color="red", label=1)
        plt.plot(X2[:, 0], X2[:, 1], '.', alpha=1, color="blue", label=2)
        plt.plot(X3[:, 0], X3[:, 1], '.', alpha=1, color="green", label=3)
        plt.plot(X4[:, 0], X4[:, 1], '.', alpha=1, color="black", label=4)
        plt.plot(X5[:, 0], X5[:, 1], '.', alpha=1, color="orange", label=5)
        plt.title("True data")
        plt.grid()
        plt.legend()
        plt.show()
    return X


def data_example_6(show_plot=True):
    X = np.empty((0, 2))
    np.random.seed(1)
    for i in range(4):
        for j in range(4):
            mu1 = np.array([i, j])
            cov1 = np.array([[0.01, 0], [0, 0.01]])
            xs = multivariate_normal.rvs(mu1, cov1, size=50, random_state=1)
            if show_plot:
                plt.plot(xs[:, 0], xs[:, 1], '.', alpha=1, color=np.random.rand(3, ))
            X = np.vstack((X, xs))
    X = np.take(X, np.random.rand(X.shape[0]).argsort(), axis=0, out=X)
    if show_plot:
        plt.grid()
        plt.legend()
        plt.title("True data")
        plt.show()
    return X


def data_example_7(show_plot=True):
    X = np.empty((0, 3))
    if show_plot:
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    real_means = np.array([[-5, -10, 0],
                           [0, -10, 0],
                           [5, -10, 0],
                           [-5, -0, 4],
                           [0, -0, 4],
                           [5, -0, 4],
                           [-5, 10, 8],
                           [0, 10, 8],
                           [5, 10, 8]])
    real_covs = np.array([[[.5, 0, 0], [0, 2, 0], [0, 0, 1]],
                          [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
                          [[.5, 0, 0], [0, 2, 0], [0, 0, 1]],
                          [[.5, 0, 0], [0, 8, 0], [0, 0, 2]],
                          [[1, 0, 0], [0, 8, 0], [0, 0, 2]],
                          [[.5, 0, 0], [0, 8, 0], [0, 0, 2]],
                          [[.5, 0, 0], [0, 2, 0], [0, 0, 1]],
                          [[1, 0, 0], [0, 2, 0], [0, 0, 1]],
                          [[.5, 0, 0], [0, 2, 0], [0, 0, 1]]])
    mix_prob = np.array([1 / 16, 2 / 16, 1 / 16, 2 / 16, 4 / 16, 2 / 16, 1 / 16, 2 / 16, 1 / 16])
    for i in range(len(real_means)):
        xs = multivariate_normal.rvs(real_means[i], real_covs[i], size=int(mix_prob[i] * 1600), random_state=1)
        if show_plot:
            ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2], color=np.append(np.random.rand(3, ), 0.5), cmap=plt.cm.Set1,
                       edgecolor="k", s=40, label=i + 1)
            ax.legend()
        X = np.vstack((X, xs))
    X = np.take(X, np.random.rand(X.shape[0]).argsort(), axis=0, out=X)
    if show_plot:
        ax.legend()
        plt.grid()
        plt.legend()
        plt.title("True data")
        plt.show()
    return X
