# -*- coding: utf-8 -*-
"""
Gaussian Mixture model
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

np.random.seed(1)
alpha1 = 0.5
alpha2 = 0.5
mu1 = np.array([0,0])
mu2 = np.array([20, 0])
cov1 = np.array([[1, 0],[0, 1]])
cov2 = np.array([[9, 0],[0, 9]])

X1 = multivariate_normal.rvs(mu1, cov1, size = 400, random_state=1)
X2 = multivariate_normal.rvs(mu2, cov2, size = 100, random_state=1)
X = np.vstack((X1, X2))
X = np.take(X,np.random.rand(X.shape[0]).argsort(),axis=0,out=X)

plt.plot(X1[:, 0], X1[:, 1], '.', alpha=1, color = "red")
plt.plot(X2[:, 0], X2[:, 1], '.', alpha=1, color = "blue")
plt.ylim(-8, 8)
plt.grid()
plt.show()


n = 2
n_iter = 10000
eps = 0.01
alphas = np.ones(n)/n
mus = X[np.random.choice(X.shape[0], n, replace = False)]
covs = np.array([np.eye(n) for i in range(n)])
#initializing zs - E step
zs = np.zeros((X.shape[0], n))
for i in range(n):
    zs[:, i] = alphas[i] * multivariate_normal.pdf(X, mean = mus[i], cov = covs[i])
zs = zs/np.sum(zs, axis = 1, keepdims = True)
zs_prev = zs   
for _ in range(n_iter): 
    #alphas
    alphas = np.sum(zs, axis = 0)/X.shape[0]   
    #means
    mus = np.dot(zs.T, X)/(np.sum(zs, axis = 0).reshape(-1, 1))
    
    for i in range(n):
        vec = X - mus[i]
        
        covs[i] = np.dot(zs[:, i]* vec.T, vec)/np.sum(zs, axis = 0)
    
    for i in range(n):
        zs[:, i] = alphas[i] * multivariate_normal.pdf(X, mean = mus[i], cov = covs[i])
    zs = zs/np.sum(zs, axis = 1, keepdims = True)
    
    if np.linalg.norm(zs - zs_prev) < eps:
        break
    else:
        zs_prev = zs
    
    

def plot_contours(data, means, covs):
    
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], '.', alpha=1, color = "black")

    
    k = means.shape[0]
    x = np.linspace(-10, 30, 1000)
    y = np.linspace(-8, 8, 1000)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    col = ['green', 'red']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal.pdf(coordinates, mean, cov).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors = col[i])
    plt.ylim(-8, 8)
    plt.grid()
    plt.show()

plot_contours(X, mus, covs)

