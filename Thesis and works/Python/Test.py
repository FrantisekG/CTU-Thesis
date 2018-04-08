# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:23:46 2018

@author: FG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#from keras.models import Sequential
#from keras.layers import Dense, LSTM
#from sklearn.preprocessing import MinMaxScaler


price=pd.read_csv("C:/Users/FG/Desktop/BTC-USD.csv",parse_dates=True,index_col=0)
#price = np.round(price["Close"].rolling(window = 20, center = False).mean(), 3).plot()
plt.plot()
price.plot(grid=True)

#mean=pd.rolling_mean(price["Close"],10)
#price.plot(all_y=["Date","Open", "High", "Low", "Close", "Adj Close", "Volume",grid = True)
#price["Close"].plot()
#plt.plot()
#mean=[price["Close"].mean()]
#print(mean)
##p=pd.DataFrame(price["Close"])
#price.head()
#def main(argv):
 

    #list or data needs to be of the same length for plotting
#if __name__ == '__main__':
  #  main(sys.argv[1:])

