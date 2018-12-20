# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:13:49 2018

@author: FG
"""

import pandas as pd
import matplotlib.pyplot as plt
#Choosing parametrs by users
Columns=[]
Window_size=[]
Num_rows=[]

df=pd.read_csv("C:/Users/FG/Desktop/Skola/Bakalarka/Thesis and works/BTC-USD.csv",parse_dates=True,nrows=200,usecols=["Close"])
df_ewma = df.ewm(com=0.5).mean() #(halflife=1 - np.log(2)/3).mean()
plt.plot(df,color="b")
plt.plot(df_ewma,color="r")
plt.ylabel("Price")
plt.title("Moving Average")

