# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:01:04 2022

@author: aleks
"""

from generate_data import *
from robust_gmm import RobustGMM
X = data_example_6()
rgmm = RobustGMM()
rgmm.fit(X)
preds = rgmm.predict(X)
rgmm.make_clusters()
rgmm.plot_predictions()