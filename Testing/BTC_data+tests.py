
# coding: utf-8

# In[9]:

import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
#locating the dataset
#boolean or list of ints or names or list of lists or dict, default False
btc_price=pd.read_csv("Desktop/coinbaseUSD_1-min_data_2014-12-01_to_2018-01-08.csv",parse_dates=True,index_col=0)
btc_price["Close"].plot()


# In[ ]:




# In[ ]:



