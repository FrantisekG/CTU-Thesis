# -*- coding: utf-8 -*-
"""
Moving Average skript

@author: FG
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Choosing parametrs by users
columns=["Close"]
window_size=3
num_rows=100

df=pd.read_csv("C:/Users/FG/Desktop/Skola/Bakalarka/Thesis and works/BTC-USD.csv",parse_dates=True,nrows=num_rows,usecols=columns)
df_mean=df.rolling(window_size).mean()
plt.plot(df_mean,color="r")
plt.plot(df,color="b")
plt.ylabel("Closing Price")
plt.title("Moving Average")


