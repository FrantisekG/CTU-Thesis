# -*- coding: utf-8 -*-
"""
Printing a graph with a Moving Average. 
Moving average, window size and use of columns is adjustable
!! How to specify and print into graph Date !!

@author: FG
"""
import pandas as pd
import matplotlib.pyplot as plt
#usecols for specifing col strings, nrows for number of rows

prc=pd.read_csv("C:/Users/FG/Desktop/BTC-USD.csv",parse_dates=True, nrows=50, usecols=["Close"])
# method for moving average
prc_ma=prc.rolling(5).mean()
#rolling finding
#prc_sub=prc.loc["2013-05-05":"2015-05-06"]

#plotting 
#figure, more advanced form of plotting

plt.plot(prc, color="blue", label="Price")
plt.plot(prc_ma, color="red", label="Moving Average")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Moving Average")
plt.grid()

