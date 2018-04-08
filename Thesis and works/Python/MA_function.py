# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 15:40:48 2018

@author: FG
"""
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/FG/Desktop/Skola/Bakalářka/Thesis and works/Python/coindesk-bpi-USD-close_data-2010-07-18_2018-04-06.csv",parse_dates=True,nrows=500)
#Moving Average  
def MA(df, n):  
    MA = pd.Series(pd.rolling_mean(df['Close Price'], n), name = 'MA_' + str(n))  
    df = df.join(MA)
    return df
