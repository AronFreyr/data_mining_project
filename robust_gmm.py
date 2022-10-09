# -*- coding: utf-8 -*-
"""
Robust Gaussian Mixture Model clustering
"""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import silhouette_score
from mpl_toolkits import mplot3d

import generate_graphs


class RobustGMM:
    
    def __init__(self, c=None, eps=0.001, gamma=0.0001, max_iter=1000,
                 plot_intermediate_steps=False, plot_step_counter=1):
        self.c = c
        self.eps = eps
        self.gamma = gamma
        self.max_iter = max_iter
        self.beta = 1
        self.plot_intermediate_steps = plot_intermediate_steps
        self.plot_step_counter = plot_step_counter  # This exists because Pycharm won't generate more than 30 plots.
        
    def init_params(self, X):
        self.X = X
        
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        if self.c is None:
            self.c = X.shape[0]
            self.mus = np.copy(self.X)
        else:
            self.mus = np.copy(self.X[np.random.choice(self.n, self.c, replace=False)])
        self.alphas = np.ones(self.c)/self.c
        self.cs = [self.c]
        self.zs = np.zeros((self.n, self.c))
        self.if_beta = True
        
    def fit(self, X):
        # step 1
        self.init_params(X)
        # step 2
        self.init_covariance()
        # step 3
        self.update_zs()
        # step 4
        self.update_mus()
        for w in range(self.max_iter):
            self.w = w
            # step 5
            self.update_alphas()
            # step 6
            self.update_beta()
            # step 7
            self.update_c()
            # step 8
            self.update_covs()
            # step 9
            self.update_zs()
            # step 10
            mus_prev = np.copy(self.mus)
            self.update_mus()

            # Generate plots
            if self.plot_intermediate_steps:
                if w % self.plot_step_counter == 0:
                    plot_title = 'Clustering results after iteration ' + str(w)
                    generate_graphs.plot_predictions(self.d, self.X, self.c, self.make_clusters(), plot_title=plot_title)

            # step 11
            if max([np.linalg.norm(self.mus[k] - mus_prev[k]) for k in range(self.c)]) < self.eps:
                break
            
    def init_covariance(self):
        Dks = np.zeros((self.c, self.n))
        for k in range(self.c):
            for i in range(self.n):             
                Dks[k][i] = np.linalg.norm(self.X[i] - self.mus[k])**2
            Dks[k] = np.sort(Dks[k])
        self.covs = np.array([Dks[k][Dks[k] > 0][int(np.sqrt(self.c))] * np.eye(self.d) for k in range(self.c)])
        self.Q = np.min(Dks[Dks > 0]) * np.eye(self.d)
        
    def update_zs(self):     
        for i in range(self.c):
            self.zs[:, i] = self.alphas[i] * multivariate_normal.pdf(self.X, mean=self.mus[i], cov=self.covs[i])
        self.zs = self.zs/np.sum(self.zs, axis=1, keepdims=True)
    
    def update_mus(self):
        self.mus = np.dot(self.zs.T, self.X)/(np.sum(self.zs, axis=0)).reshape(-1, 1)
    
    def update_alphas(self):
        self.alphas_em = np.sum(self.zs, axis=0)/self.n
        self.alphas_old = np.copy(self.alphas)
        self.E = np.sum(self.alphas_old*np.log(self.alphas_old))
        self.alphas = self.alphas_em + self.beta*self.alphas_old*(np.log(self.alphas_old) - self.E)
        
    def update_beta(self):
        if self.if_beta:
            power = np.trunc(self.d/2 - 1)
            eta = min(1, 0.5 ** power)
            v1 = np.sum(np.exp(-eta*self.n*np.abs(self.alphas - self.alphas_old)))/self.c
            v2 = (1 - max(self.alphas_em))/(-max(self.alphas_old)*self.E)
            self.beta = min(v1, v2)
    
    def update_c(self):
        self.idx = self.alphas < 1/self.n
        size = len(self.alphas[self.alphas < 1/self.n])
        self.c = self.c - size
        self.cs.append(self.c)
        self.alphas = self.alphas[~self.idx]/np.sum(self.alphas[~self.idx])
        self.zs = self.zs[:, ~self.idx]/np.sum(self.zs[:, ~self.idx], axis=1, keepdims=True)
        
        if self.w >= 60 and self.cs[self.w-60] - self.cs[self.w] == 0:
            self.beta = 0
            self.if_beta = False
            
    def update_covs(self):
        self.mus = self.mus[~self.idx]
        arr = []
        for i in range(self.c):
            vec = self.X - self.mus[i]          
            # 26
            self.covs[i] = np.dot(self.zs[:, i]*vec.T, vec)/np.sum(self.zs, axis=0)[i]
            # 28
            self.covs[i] = ((1-self.gamma)*self.covs[i]) + (self.gamma*self.Q)
            arr.append(np.copy(self.covs[i]))
        self.covs = np.array(arr)
        
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
        generate_graphs.plot_predictions(self.d, self.X, self.c, self.clusters)
