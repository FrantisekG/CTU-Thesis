
# coding: utf-8

# In[ ]:

import time
import sys
import datetime

from poloniex import Poloniex

polo = poloniex.Poloniex('Public key','Secret key')
balance = polo('returnBalances')
print("I have %s BTC!" % balance['BTC'])

#def main(argv):
#    period = 10 
    

        

