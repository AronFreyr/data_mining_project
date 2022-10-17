# -*- coding: utf-8 -*-
"""
Gaussian Mixture model
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import silhouette_score

import generate_graphs


class GMM:

    def __init__(self, n, mus=None, covs=None, n_iter=1000, eps=0.1):
        self.n = n  # number of clusters
        self.n_iter = n_iter  # number of iterations
        self.eps = eps  # epsilon
        self.mus = mus  # mus array (n x k)
        self.covs = covs  # covs array (n x k x k)

        self.X = None  # Initialized in init_params
        self.dim = None  # Initialized in init_params
        self.alphas = None  # Initialized in init_params
        self.clusters = None  # Initialized in init_params
        self.zs = None  # Initialized in init_zs

    def init_params(self, X):
        # get data
        self.X = X
        # get dimension of data
        self.dim = self.X.shape[1]
        # set alphas to 1/n
        self.alphas = np.ones(self.n)/self.n
        if self.mus is None:
            # mus - random n rows of X
            self.mus = X[np.random.choice(self.X.shape[0], self.n, replace=False)]
        if self.covs is None:
            # eye matrices
            self.covs = np.array([np.eye(self.dim) for _ in range(self.n)])
     
    def init_zs(self):
        # init z to init alphas
        self.zs = np.zeros((self.X.shape[0], self.n))
        for i in range(self.n):
            self.zs[:, i] = self.alphas[i] * multivariate_normal.pdf(self.X, mean=self.mus[i], cov=self.covs[i])
        self.zs = self.zs/np.sum(self.zs, axis=1, keepdims=True)
    
    def update_zs(self):
        # eq 4
        for i in range(self.n):
            self.zs[:, i] = self.alphas[i] * multivariate_normal.pdf(self.X, mean=self.mus[i], cov=self.covs[i])
        self.zs = self.zs/np.sum(self.zs, axis=1, keepdims=True)

    def compute_alpha(self):
        # eq 6
        self.alphas = np.sum(self.zs, axis=0)/self.X.shape[0]
        
    def compute_mu(self):
        # eq 8
        self.mus = np.dot(self.zs.T, self.X)/(np.sum(self.zs, axis=0)).reshape(-1, 1)
    
    def compute_cov(self):
        # eq 9
        for i in range(self.n):
            vec = self.X - self.mus[i]          
            self.covs[i] = np.dot(self.zs[:, i]*vec.T, vec)/np.sum(self.zs, axis=0)[i]
            
    def fit(self, X):
        # algorithm
        self.init_params(X)
        self.init_zs()
        zs_prev = np.copy(self.zs)
        for _ in range(self.n_iter):
            self.compute_alpha()
            self.compute_mu()
            self.compute_cov()
            self.update_zs()
            zs_prev = np.copy(self.zs)
            # step 4 - break condition
            if np.linalg.norm(self.zs - zs_prev) < self.eps:
                break
            
    def make_clusters(self):
        # make clusters for data provided
        # zs are probabilities of belonging to each class
        self.clusters = np.argmax(self.zs, axis=1)
        return self.clusters
    
    def predict(self, Y):
        # predict on unseen data
        # find to which cluster Mahalenobis distance is minimized
        dists = []
        for p in range(Y.shape[0]):
            lista = []
            for i in range(len(self.mus)):
                delta = Y[p] - self.mus[i]
                m = np.dot(np.dot(delta, np.linalg.inv(self.covs[i])), delta)
                lista.append(np.sqrt(m))
            dists.append(np.argmin(lista))
        return np.array(dists)
     
    def plot_predictions(self):
        generate_graphs.plot_predictions(self.dim, self.X, self.predict(self.X), self.clusters)

    def silhouette(self):
        return silhouette_score(self.X, self.clusters)
