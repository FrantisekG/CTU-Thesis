# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 23:11:09 2018

@author: FG
"""

import csv
import time as tm
import datetime as dt

with open("BTC-USD.csv") as csvfile:
    btc_data=list(csv.DictReader(csvfile))
    
btc_data[:10]

len(btc_data)

btc_data[0].keys()

tm.time()