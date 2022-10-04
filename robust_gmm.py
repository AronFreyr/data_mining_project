# -*- coding: utf-8 -*-
"""
Robust Gaussian Mixture Model clustering
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import silhouette_score
from mpl_toolkits import mplot3d

def generate_data_for_gmm(show_plot=True):
    # generating data
    np.random.seed(1)
    alpha1 = 0.5
    alpha2 = 0.5
    mu1 = np.array([0, 0])
    mu2 = np.array([20, 0])
    cov1 = np.array([[1, 0], [0, 1]])
    cov2 = np.array([[9, 0], [0, 9]])

    X1 = multivariate_normal.rvs(mu1, cov1, size=400, random_state=1)
    X2 = multivariate_normal.rvs(mu2, cov2, size=100, random_state=1)
    X = np.vstack((X1, X2))
    X = np.take(X, np.random.rand(X.shape[0]).argsort(), axis=0, out=X)
    try:
        plt.plot(X1[:, 0], X1[:, 1], '.', alpha=1, color="red")
        plt.plot(X2[:, 0], X2[:, 1], '.', alpha=1, color="blue")
    except AttributeError:
        # Fix for the error: AttributeError: module 'backend_interagg' has no attribute 'FigureCanvas'
        matplotlib.use('TkAgg')
        plt.plot(X1[:, 0], X1[:, 1], '.', alpha=1, color="red")
        plt.plot(X2[:, 0], X2[:, 1], '.', alpha=1, color="blue")

    plt.ylim(-8, 8)
    plt.grid()
    if show_plot:
        plt.show()
    return X
X = generate_data_for_gmm()

#step 1


eps = 0.01
beta = 1
c = X.shape[0]     #TODO make c input, if c is None then c is n
d = X.shape[1]
alphas = np.ones(c)/c
mus = np.copy(X[np.random.choice(X.shape[0], c, replace=False)])

#eq 27
Dks = np.zeros((c, X.shape[0]))
for k in range(c):
    for i in range(X.shape[0]):
        if i != k:
            Dks[k][i] = np.linalg.norm(X[i] - mus[k])**2
    Dks[k] = np.sort(Dks[k])[::-1]
            
covs = np.array([Dks[k][Dks[k] > 0][int(np.sqrt(c))] * np.eye(d) for k in range(c)])

#STEP 3
zs = np.zeros((X.shape[0], c))
for i in range(c):
    zs[:, i] = alphas[i] * multivariate_normal.pdf(X, mean=mus[i], cov=covs[i])
zs = zs/np.sum(zs, axis=1, keepdims=True)

#STEP 4
mus = np.dot(zs.T, X)/(np.sum(zs, axis=0)).reshape(-1, 1)

#STEP 5
alphas_em = np.sum(zs, axis = 0)/X.shape[0]
alphas_old = np.copy(alphas)
E = np.sum(alphas_old*np.log(alphas_old))
alphas = alphas_em + beta*alphas_old*(np.log(alphas_old) - E)

#STEP 6
power = np.trunc(d/2 - 1)
eta = min(1, power)
v1 = np.sum(np.exp(-eta*(alphas - alphas_old)))/c
v2 = (1 - max(alphas_em))/(-max(alphas_old)*E)
beta = min(v1, v2)




