# -*- coding: utf-8 -*-
"""
Neural network
nazev souboru, skip rows, 
všechny nutné proměnné dát na začátek
@author: FG
"""
import numpy as np
import pandas as pd
#import talib
#import random
#random.seed(42)
import matplotlib.pyplot as plt
#import 
prc=pd.read_csv("C:/Users/FG/Desktop/BTC-USD.csv", parse_dates=True, nrows=60)
#preparation
prc=prc[['Open', 'High', 'Low', 'Close']]
prc['H-L'] = prc['High'] - prc['Low']
prc['O-C'] = prc['Close'] - prc['Open']
prc_ma=prc.rolling(3).mean()#5 day rolling average
prc_std=prc['Close'].rolling(3).std()#Standard devatin

prc['Price_Rise'] = np.where(prc['Close'].shift(-1) > prc['Close'], 1, 0)
#select rows and columns by number, in the order that they appear in the data frame.
X = prc.iloc[:, 4:-1]
y = prc.iloc[:, -1]

plt.plot(prc, color="blue", label="Price")
plt.plot(prc_ma, color="red", label="Moving Average")
plt.plot(prc_std, color="green", label="Standard devation")
plt.plot(X, color="black")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Moving Average, Standard Devation")
plt.grid()

#slovnik a dat for pro iteraci objektů
#[for lin in lines_to plot]

#Splitting the dataset
split = int(len(prc)*0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

