# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:01:04 2022

@author: aleks
"""

from generate_data import *
from robust_gmm import RobustGMM
import numpy as np
X = data_example_5(show_plot = False)
rgmm = RobustGMM()
rgmm.fit(X)
rgmm.make_clusters()
rgmm.plot_predictions()



c = X.shape[0] - 100
n = X.shape[0]
d = X.shape[1]
mus = np.copy(X[np.random.choice(n, c, replace=False)])
Dks = np.zeros((c, n))
for k in range(c):
    for i in range(n):
        
        Dks[k][i] = np.linalg.norm(X[i] - mus[k])**2
        
        
    Dks[k] = np.sort(Dks[k])
 
    
 
#until now works for sure!!!!!!!!!!
covs = np.array([Dks[k][Dks[k] > 0][int(np.sqrt(c))] * np.eye(d) for k in range(c)])
Q = np.min(Dks[Dks > 0])* np.eye(d)

#It works. Great success.

def _initialize_covmat_1d(d_k):
        """
        self._initialize_covmat() that uses np.apply_along_axis().
        This function is refered term 27 in the paper.
        Args:
            d_k: numpy 1d array
        """
        d_k = d_k.copy()
        d_k.sort()
        d_k = d_k[d_k != 0]
        return ((d_k[int(np.ceil(np.sqrt(c)))]) * np.identity(d))
    
    
#second norm squared
D_mat = np.sum((X[None, :]-mus[:, None])**2, -1)
#sort rows
D_mat = np.array([np.sort(D_mat[k]) for k in range(c)])


covs2 = np.apply_along_axis(
            func1d=lambda x: _initialize_covmat_1d(x),
            axis=1, arr=D_mat)
D_mat_reshape = D_mat.reshape(-1, 1)
d_min = D_mat_reshape[D_mat_reshape > 0].min()
Q2 = d_min*np.identity(d)