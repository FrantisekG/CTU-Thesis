# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 16:55:58 2018

@author: FG
"""

import numpy as np
from TFANN import ANNR
import matplotlib.pyplot as mpl
from sklearn.preprocessing import scale

pth = ("C:/Users/FG/Desktop/BTC-USD.csv")
A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(1, 4))
A = scale(A)
#y is the dependent variable
y = A[:, 1].reshape(-1, 1)
#A contains the independent variable
A = A[:, 0].reshape(-1, 1)
#Plot the high value of the stock price
mpl.plot(A[:, 0], y[:, 0])
mpl.show()