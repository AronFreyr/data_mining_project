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

#step 1
class RobustGMM:
    
    def __init__(self, c = None, eps = 0.001, gamma = 0.0001, max_iter = 1000):
        self.c = c
        self.eps = eps
        self.gamma = gamma
        self.max_iter = max_iter
        self.beta = 1
        
    def init_params(self, X):
        self.X = X
        if self.c is None:
            self.c = X.shape[0]
        self.n = self.X.shape[0]
        self.d = self.X.shape[1]
        self.alphas = np.ones(self.c)/self.c
        self.mus = np.copy(self.X[np.random.choice(self.n, self.c, replace=False)])
        self.cs = [self.c]
        self.zs = np.zeros((self.n, self.c))
        
    def fit(self, X):
        #step 1
        self.init_params(X)
        #step 2
        self.init_covariance()
        #step 3
        self.update_zs()
        #step 4
        self.update_mus()
        for w in range(self.max_iter):
            self.w = w
            #step 5
            self.update_alphas()
            #step 6
            self.update_beta()
            #step 7
            self.update_c()
            #step 8
            self.update_covs()
            #step 9
            self.update_zs()
            #step 10
            mus_prev = np.copy(self.mus)
            self.update_mus()
            #step 11
            if max([np.linalg.norm(self.mus[k] - mus_prev[k]) for k in range(self.c)]) < self.eps:
                break
            
    def init_covariance(self):
        Dks = np.zeros((self.c, self.n))
        for k in range(self.c):
            for i in range(self.n):
                if i != k:
                    Dks[k][i] = np.linalg.norm(self.X[i] - self.mus[k])**2
            Dks[k] = np.sort(Dks[k])[::-1]
        self.covs = np.array([Dks[k][Dks[k] > 0][int(np.sqrt(self.c))] * np.eye(self.d) for k in range(self.c)])
        self.Q = np.min(Dks[Dks > 0])* np.eye(self.d)
        
    def update_zs(self):     
        for i in range(self.c):
            self.zs[:, i] = self.alphas[i] * multivariate_normal.pdf(self.X, mean=self.mus[i], cov=self.covs[i])
        self.zs = self.zs/np.sum(self.zs, axis=1, keepdims=True)
    
    def update_mus(self):
        self.mus = np.dot(self.zs.T, self.X)/(np.sum(self.zs, axis=0)).reshape(-1, 1)
    
    def update_alphas(self):
        self.alphas_em = np.sum(self.zs, axis = 0)/self.n
        self.alphas_old = np.copy(self.alphas)
        self.E = np.sum(self.alphas_old*np.log(self.alphas_old))
        self.alphas = self.alphas_em + self.beta*self.alphas_old*(np.log(self.alphas_old) - self.E)
        
    def update_beta(self):
        power = np.trunc(self.d/2 - 1)
        eta = min(1, power)
        v1 = np.sum(np.exp(-eta*(self.alphas - self.alphas_old)))/self.c
        v2 = (1 - max(self.alphas_em))/(-max(self.alphas_old)*self.E)
        self.beta = min(v1, v2)
    
    def update_c(self):
        self.idx = self.alphas < 1/self.n
        size = len(self.alphas[self.alphas < 1/self.n])
        self.c = self.c - size
        self.cs.append(self.c)
        self.alphas = self.alphas[~self.idx]/np.sum(self.alphas[~self.idx])
        self.zs = self.zs[:, ~self.idx]/np.sum(self.zs[:, ~self.idx], axis = 1, keepdims = True)
        
        if self.w >= 60 and self.c[self.w-60] - self.c[self.w] == 0:
            self.beta = 0
            
    def update_covs(self):
        self.mus = self.mus[~self.idx]
        arr = []
        for i in range(self.c):
            vec = self.X - self.mus[i]          
            #26
            self.covs[i] = np.dot(self.zs[:, i]*vec.T, vec)/np.sum(self.zs, axis=0)[i]
            #28
            self.covs[i] = ((1-self.gamma)*self.covs[i]) + (self.gamma*self.Q)
            arr.append(np.copy(self.covs[i]))
        self.covs = np.array(arr)
        
    def make_clusters(self):
        # make clusters for data provided
        # zs are probabilities of belonging to each class
        self.clusters = np.argmax(self.zs, axis=1)
        return self.clusters
    
    def predict(self, Y):
        #predict on unseen data
        #find to which cluster Mahalenobis distance is minimized
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
        if self.d <= 3:
            if self.d == 1:
                X_with_preds = np.c_[self.X, self.clusters]
                colors = [np.random.rand(3,) for _ in range(self.n)]
                for i in range(self.X.shape[0]):
                    plt.plot(self.X[i], '.', alpha=1, color=colors[int(self.X_with_preds[i][1])])
                plt.grid()
                plt.tick_params(
                    axis='x',          
                    which='both',     
                    bottom=False,      
                    top=False,         
                    labelbottom=False)
                plt.show()
                    
            if self.d == 2:
                for i, cluster in enumerate(np.unique(self.clusters)):
                    c = self.X[self.clusters == i]
                    plt.plot(c[:, 0], c[:, 1], '.', alpha=1, color=np.random.rand(3,))
                plt.grid()
                plt.show()

            if self.d == 3:
                ax = plt.axes(projection='3d')
                for i, cluster in enumerate(np.unique(self.clusters)):
                    c = self.X[self.clusters == i]
                    ax.scatter3D(c[:, 0], c[:, 1],  c[:, 2], alpha=1, color=np.random.rand(3,))
                plt.grid()
                plt.show()
        else:
            raise ValueError("This method can only plot 1-3D data")













