# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:01:04 2022

@author: aleks
"""

from generate_data import *
from robust_gmm import RobustGMM
import numpy as np
from sklearn.datasets import make_moons, load_iris
from sklearn.decomposition import PCA


X = data_example_3(show_plot = True)
rgmm = RobustGMM()
rgmm.fit(X)
rgmm.make_clusters()
rgmm.plot_predictions()

#moons
moons = make_moons(n_samples=1000)
rgmm = RobustGMM()
rgmm.fit(moons[0])
rgmm.make_clusters()
rgmm.plot_predictions()


#iris
iris = load_iris()
rgmm = RobustGMM()
rgmm.fit(iris.data)
clusters = rgmm.make_clusters()

#iris dim reduction
X_reduced = PCA(n_components=2).fit_transform(iris.data)
fig = plt.figure(1, figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=iris.target, edgecolor="k")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")


rgmm = RobustGMM()
rgmm.fit(X_reduced)
clusters = rgmm.make_clusters()
rgmm.plot_predictions()

X = data_example_1_3D(show_plot = True)
rgmm = RobustGMM()
rgmm.fit(X)
rgmm.make_clusters()
rgmm.plot_predictions()