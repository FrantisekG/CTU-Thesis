
# coding: utf-8

# In[9]:
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
#locating the dataset
#boolean or list of ints or names or list of lists or dict, default False
btc_price=pd.read_csv("Desktop/BTC-USD.csv",parse_dates=True,index_col=0)
btc_price["Close"]["2016-12-31":"2017-12-31"].plot()
pd.rolling_mean(btc_price["Close"]["2016-12-31":"2017-12-31"],5).plot()
