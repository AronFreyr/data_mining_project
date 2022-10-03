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



class GMM:
    def __init__(self, n, n_iter = 1000, eps = 0.1):
        self.n = n
        self.n_iter = n_iter
        self.eps = eps
        
    def init_params(self, X):
        self.X = X
        dim = self.X.shape[1]
        self.alphas = np.ones(self.n)/self.n
        #mus - random n rows of X
        self.mus = X[np.random.choice(self.X.shape[0], self.n, replace = False)]
        self.covs = np.array([np.eye(dim) for _ in range(self.n)])
    
    def update_zs(self):
        for i in range(self.n):
            self.zs[:, i] = self.alphas[i] * multivariate_normal.pdf(X, mean = self.mus[i], cov = self.covs[i])
        self.zs = self.zs/np.sum(self.zs, axis = 1, keepdims = True)
        
    def init_zs(self):
        self.zs = np.zeros((self.X.shape[0], self.n))
        for i in range(self.n):
            self.zs[:, i] = self.alphas[i] * multivariate_normal.pdf(X, mean = self.mus[i], cov = self.covs[i])
        self.zs = self.zs/np.sum(self.zs, axis = 1, keepdims = True)
        
        
    def compute_alpha(self):
        self.alphas = np.sum(self.zs, axis = 0)/self.X.shape[0]  
        
    def compute_mu(self):
        self.mus = np.dot(self.zs.T, X)/(np.sum(self.zs, axis = 0)).reshape(-1, 1)
    
    def compute_cov(self):
        for i in range(self.n):
            vec = self.X - self.mus[i]          
            self.covs[i] = np.dot(self.zs[:, i]* vec.T, vec)/np.sum(self.zs, axis = 0)[i]
            
    def fit(self, X):
        self.init_params(X)
        self.init_zs()
        zs_prev = np.copy(self.zs)
        for _ in range(self.n_iter):
            self.compute_alpha()
            self.compute_mu()
            self.compute_cov()
            self.update_zs()
            zs_prev = np.copy(self.zs)
            if np.linalg.norm(self.zs - zs_prev) < self.eps:
                break
            
    def predict(self):
        self.clusters = np.argmax(self.zs, axis = 1)
        return self.clusters
     
    def plot_predictions(self):
        for i, cluster in enumerate(np.unique(self.clusters)):
            c = self.X[preds == i]
            plt.plot(c[:, 0], c[:, 1], '.', alpha=1, color = np.random.rand(3,))
        plt.grid()
        plt.show()
    
        
        
gmm = GMM(n = 10)
gmm.fit(X)

preds = gmm.predict()
gmm.plot_predictions()











