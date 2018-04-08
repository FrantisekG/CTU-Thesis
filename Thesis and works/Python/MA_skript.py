# -*- coding: utf-8 -*-
"""
Moving Average skript

@author: FG
"""

import pandas as pd
import matplotlib.pyplot as plt
# User inputs
#Date,Open,High,Low,Close,Adj Close,Volume
Column=[]
#Choosing a row
Number_of_rows=50
#MA window size
MA_window_size=[]

"""
Skript section
"""
df=pd.read_csv("C:/Users/FG/Desktop/Skola/Bakalářka/Thesis and works/Python/coindesk-bpi-USD-close_data-2010-07-18_2018-04-06.csv",parse_dates=True,nrows=500,usecols=["Close Price"])
df_mean=df.rolling(15).mean()
plt.plot(df_mean,color="r")
plt.plot(df,color="b")
plt.grid()

