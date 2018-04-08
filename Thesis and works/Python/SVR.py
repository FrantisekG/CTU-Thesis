# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:01:37 2018

@author: FG
"""
import csv
import numpy as np
from sklearn.svm import SVR
import pandas as pd
import matplotlib.pyplot as plt #depends on a graphical backend
#if it doesnt want to plot, try out switch_backend
#plt.switch_backend("new_backend") #function


dates=[]
prices=[]

def get_data(filename):
    with open("BTC-USD.csv","r") as csvfile: # r parameter for reading
             csvFileReader=csv.reader(csvfile) #variable and reader method
             next(csvFileReader)#itarates every row but skips the first row: Date, Open etc...
             for row in csvFileReader:#return a string for each line
                 #removes the - from dates and turnes day into an int?
                 dates.append(int(row[0].split("-")[0]))
                 #append function allows us to ad a item to the data ; adding split to remove dash from date
                 prices.append(float(row[1])) #converting first row to float
                 #return
                 # keeping the function unfinished

        

def predict_prices(dates,prices, x): #for forming with the help of numpy into a 4x1 matrixs
        dates=np.reshape(dates,(len(dates),1))
        
        svr_lin = SVR(kernel="linear")
        svr_poly = SVR(kernel="poly", degree=2)
        svr_rbf=SVR(kernel='rbf', gamma=1)
        svr_lin.fit(dates,prices)
        svr_poly.fit(dates, prices)
        svr_rbf.fit(dates,prices)
        
        plt.scatter(dates, prices, color="black", label="Data")
        plt.plot(dates, svr_rbf.predict(dates), color="red", label="RBF model")
        plt.plot(dates, svr_lin.predict(dates), color="green", label="Linear model")
        plt.plot(dates, svr_poly.predict(dates), color="blue", label="Polynomial model")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Support Vector Regression")
        plt.grid()
        plt.legend()
        plt.show()
        
        return svr_rbf.predict(x)[0],svr_lin.predict(x)[0],svr_poly.predict(x)[0] 

get_data("BTC-USD.csv")

predicted_price=predict_prices(dates,prices,30)

print(predicted_price)
     
                 
    
    