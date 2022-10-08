# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:01:04 2022

@author: aleks
"""

from generate_data import *
from robust_gmm import RobustGMM
import numpy as np
X = data_example_5(show_plot = True)
rgmm = RobustGMM()
rgmm.fit(X)
rgmm.make_clusters()
rgmm.plot_predictions()

