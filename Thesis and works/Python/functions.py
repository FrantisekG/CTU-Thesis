# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 17:43:09 2018

@author: FG
"""

def multi(a,b):
    if b==1:
        return a
    else:
        return a + mult(a,b-1)
" 