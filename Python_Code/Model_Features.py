#!/usr/bin/env python
# coding: utf-8

# ## Data Load

# In[1]:


import talib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from talib import RSI, BBANDS, MA_Type, STOCH


# In[2]:


Path = "/Users/frantisek.grossmann/Desktop/Sets/crypto_data/LTC-USD.csv"
df = pd.read_csv(path, names=["Time", "Low", "High", "Open", "Close", "Volume"])
df.head()


# In[3]:


df.tail()


# ## Testing
# 

# In[4]:


df['15 minute'] = df['Close'].rolling(window=15).mean()
df['30 minute'] = df['Close'].rolling(window=30).mean()
df[['Close', '15 minute', '30 minute']].tail()


# ## Features construction using TA-lib

# In[5]:


close = df["Close"].values
high = df["High"].values
low = df["Low"].values

# Moving Average
df['3day MA'] = df['Close'].shift(1).rolling(window = 3).mean()
df['5day MA'] = df['Close'].shift(1).rolling(window = 5).mean()
df['10day MA'] = df['Close'].shift(1).rolling(window = 10).mean()
# Standard deviation
df['STD']= df['Close'].rolling(5).std()
# Relative Strength index
df['RSI'] = RSI(df['Close'].values, timeperiod = 9)
# Bollinger Bands
up, mid, low = BBANDS(close, timeperiod = 10, nbdevup = 10, nbdevdn = 10, matype = 5)
# Stochastic 
slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=10, slowk_matype=0, slowd_period=10, slowd_matype=0)


# ## Plotting 

# In[6]:


# Moving Average
#fig, ax = plt.subplots(1, figsize=(10, 6), dpi = 100)
plt.figure(figsize=(10,6), dpi=100, facecolor='w', edgecolor='k')
Slice = df.iloc[0:600, 4:5] # first thousand rows of the "Close" column
df['15 Minute MA'] = Slice.shift(1).rolling(window = 15).mean()
df['30 Minute MA'] = Slice.shift(1).rolling(window = 30).mean()
df['60 Minute MA'] = Slice.shift(1).rolling(window = 60).mean()
plt.title('Moving Average Feature against Real Closing Price', fontsize=16)
plt.plot(Slice,color="b")
plt.plot(df['15 Minute MA'],color="r")
plt.plot(df['30 Minute MA'],color="g")
plt.plot(df['60 Minute MA'],color="y")
plt.xlabel('Time [Minutes]', fontsize=14)
plt.ylabel('Closing price of currency pair LTC/USD', fontsize=12)
plt.legend(["Real Closing Price", "15 Minute Moving Average", "30 Minute Moving Average", "60 Minute Moving Average"], loc = "best", fontsize=14)
plt.grid()
plt.show()


# In[7]:


# # STANDARD DEVIATION
plt.figure(figsize=(10,6), dpi=100, facecolor='w', edgecolor='k')
df['STD']= df['Close'].rolling(15).std()
df['STD'].plot()
plt.title('Standard Deviation Feature against Real Closing Price', fontsize=16)
plt.plot(df['STD'],color="r")
plt.xlabel('Time [Minutes]', fontsize=14)
plt.ylabel('Closing price of currency pair LTC/USD', fontsize=12)
plt.legend(["Real Closing Price", "15 Standard Deviation"], loc = "best", fontsize=14)
plt.grid()
plt.show()


# In[8]:


# RELATIVE STRENGTH INDEX
plt.figure(figsize=(12,6), dpi=100, facecolor='w', edgecolor='k')
df['RSI'] = RSI(df['Close'].values, timeperiod = 9)
plt.title('LTC/USD')
plt.plot(df['RSI'],color="r")
plt.xlabel('Timestamp')
plt.ylabel('RSI')
plt.grid()
plt.show()


# In[ ]:




