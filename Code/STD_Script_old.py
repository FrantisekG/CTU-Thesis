# -*- coding: utf-8 -*-
"""
Standard devation script

@author: FG
"""

import pandas as pd
import matplotlib.pyplot as plt
#Choosing parametrs by users
columns=["Close"]
window_size=3
Num_rows=100

df=pd.read_csv("C:/Users/FG/Desktop/Skola/Bakalarka/Thesis and works/BTC-USD.csv",parse_dates=True,nrows=Num_rows,usecols=["Close"])
df_std = pd.rolling_std(df[columns], window=window_size, ddof=0)
plt.plot(df,color="b")
plt.plot(df_std,"--",color="r")
