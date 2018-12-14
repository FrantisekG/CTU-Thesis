import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from collections import deque
import random
import time
#####################################################################
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNL
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.kerar.callbacks import ModelCheckpoint

#additional modfication of plots using seaborng
plt.style.use('seaborn-notebook') 
#####################################################################
#DATA LOADING#
path = "/Users/frantisek.grossmann/Desktop/Sets/crypto_data/LTC-USD.csv"
df = pd.read_csv(path, names=['Time', 'Low', 'High', 'Open', 'Close', 'Volume'])

Seq_Length= 60 #the last 60 minutes of pricing data for prediction
Future_Time_Predict= 3 #how many minutes/hours into the future do we predict // functions like slicing
Predicted_Pair="LTC-USD"


def classify (current_price, future_price):
    if float(future_price) > float(current_price): #if future price in training data is greater than current return an int 1
        return 1 #buy
    else:
        return 0 #sell

# BALANCING DATA

def preprocess_df(df):
    df = df.drop("Future", 1) # leave out the future calm from the learning process
    for col in df.columns:
        if col != "Target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col]=preprocessing.scale(df[col].values)
    # in case the pct_change causes an NA, it will leave it out       
    df.dropna(inplace=True)

    sequential_data = []
    prev_day = deque(maxlen=Seq_Length)

    for i in df.values:
        prev_day.append([n for n in i[:-1]])
        if len(prev_day) == Seq_Length:
            sequential_data.append([np.array(prev_day), i[-1]])

    random.shuffle(sequential_data)

    Buys = []  # list that will store our buy sequences and targets
    Sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            Sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            Buys.append([seq, target])  # it's a buy!

    random.shuffle(Buys)  # shuffle the buys
    random.shuffle(Sells)  # shuffle the sells!

    lower = min(len(Buys), len(Sells))  # what's the shorter length?

    Buys = Buys[:lower]  # make sure both lists are only up to the shortest length.
    Sells = Sells[:lower]  # make sure both lists are only up to the shortest length.

    sequential_data = Buys+Sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)
    
    return np.array(X), y


##############################################################################################################
#DATAFRAME#

main_df = pd.DataFrame() # begin empty dataframe, in order to merge dataframes
pairs = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"] #strings of used Cryptos
for pair in pairs:
    #print (ratios) for control 
    #we recreate our dataset and format it with f formats the strings using "f" short version
    dataset=f"/Users/frantisek.grossmann/Desktop/Sets/crypto_data/{pair}.csv" 
    #loading of our redifened dataset
    df=pd.read_csv(dataset,names=['Time', 'Low', 'High', 'Open', 'Close', 'Volume'])
    #print(df.head()) for control purposes
    
    #renaming of columns in the dataframe
    df.rename(columns={"Close": f"{pair}_Close","Volume": f"{pair}_Volume"}, inplace=True) 
    #setting an Index for all of the pairs in the dataframe
    df.set_index("Time", inplace=True)
    # Inside the dataframe choose only Close and Volume
    df=df[[f"{pair}_Close",f"{pair}_Volume"]] 
    
    #print(df.head()) for control 
    
    if len(main_df)==0:
        main_df=df
    else:
        main_df=main_df.join(df)
    
##############################################################################################################
main_df["Future"]=main_df[f"{Predicted_Pair}_Close"].shift(-Future_Time_Predict)
main_df["Target"]=list(map(classify, main_df[f"{Predicted_Pair}_Close"],main_df["Future"]))
  
times = sorted(main_df.index.values)  # we need to sort the validation samples. 
last_5pct = sorted(main_df.index.values)[-int(0.05*len(times))] #last 5% of time

validation_main_df=main_df[(main_df.index>=last_5pct)]
main_df=main_df[(main_df.index<last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, Buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, Buys: {validation_y.count(1)}")

##############################################################################################################

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())