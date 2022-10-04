# -*- coding: utf-8 -*-
"""
Gaussian Mixture model
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
            self.zs[:, i] = self.alphas[i] * multivariate_normal.pdf(X, mean=self.mus[i], cov=self.covs[i])
        self.zs = self.zs/np.sum(self.zs, axis=1, keepdims=True)
    
    def update_zs(self):
        # eq 4
        for i in range(self.n):
            self.zs[:, i] = self.alphas[i] * multivariate_normal.pdf(X, mean=self.mus[i], cov=self.covs[i])
        self.zs = self.zs/np.sum(self.zs, axis=1, keepdims=True)

    def compute_alpha(self):
        # eq 6
        self.alphas = np.sum(self.zs, axis=0)/self.X.shape[0]
        
    def compute_mu(self):
        # eq 8
        self.mus = np.dot(self.zs.T, X)/(np.sum(self.zs, axis=0)).reshape(-1, 1)
    
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
        if self.dim <= 3:
            if self.dim == 1:
                X_with_preds = np.c_[self.X, self.clusters]
                colors = [np.random.rand(3,) for _ in range(self.n)]
                for i in range(self.X.shape[0]):
                    plt.plot(X[i], '.', alpha=1, color=colors[int(X_with_preds[i][1])])
                plt.grid()
                plt.tick_params(
                    axis='x',          
                    which='both',     
                    bottom=False,      
                    top=False,         
                    labelbottom=False)
                plt.show()
                    
            if self.dim == 2:
                for i, cluster in enumerate(np.unique(self.clusters)):
                    c = self.X[self.clusters == i]
                    plt.plot(c[:, 0], c[:, 1], '.', alpha=1, color=np.random.rand(3,))
                plt.grid()
                plt.show()

            if self.dim == 3:
                ax = plt.axes(projection='3d')
                for i, cluster in enumerate(np.unique(self.clusters)):
                    c = self.X[self.clusters == i]
                    ax.scatter3D(c[:, 0], c[:, 1],  c[:, 2], alpha=1, color=np.random.rand(3,))
                plt.grid()
                plt.show()
        else:
            raise ValueError("This method can only plot 3D data")

    def silhouette(self):
        return silhouette_score(self.X, self.clusters)
    

if __name__ == '__main__':

    X = generate_data_for_gmm()

    gmm = GMM(n=2)
    gmm.fit(X)

    preds = gmm.make_clusters()
    preds2 = gmm.predict(X)
    gmm.plot_predictions()

    np.random.seed(1)
    alpha1 = 0.5
    alpha2 = 0.5
    mu1 = np.array([0])
    mu2 = np.array([20])
    cov1 = np.array([1])
    cov2 = np.array([9])

    X1 = multivariate_normal.rvs(mu1, cov1, size=400, random_state=1)
    X2 = multivariate_normal.rvs(mu2, cov2, size=100, random_state=1)
    X = np.concatenate((X1, X2))
    X = np.take(X, np.random.rand(X.shape[0]).argsort(), axis=0, out=X).reshape(-1, 1)

    gmm = GMM(n=2)
    gmm.fit(X)
    preds = gmm.make_clusters()
    gmm.plot_predictions()

