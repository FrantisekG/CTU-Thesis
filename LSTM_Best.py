#!/usr/bin/env python
# coding: utf-8

# In[266]:


import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation


# In[267]:


path = "/Users/frantisek.grossmann/Desktop/Sets/crypto_data/LTC-USD.csv"
df = pd.read_csv(path,names=['Time', 'Low', 'High', 'Open', 'close', 'Volume'])
df.head()


# In[268]:


def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


# In[282]:


def line_plot(line1, line2, label1=None, label2=None, title=''):
    graf = fig, ax = plt.subplots(1, figsize=(10, 6), dpi = 100)
    ax.plot(line1, label=label1, linewidth=2)
    ax.plot(line2, label=label2, linewidth=2)
    ax.grid(True)
    ax.set_ylabel('Closing price of currency pair LTC/USD', fontsize=12)
    ax.set_xlabel('Time [600 Minutes]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=14)


# In[270]:


train, test = train_test_split(df, test_size=0.2)
#line_plot(train.close, test.close, 'Training', 'Test', 'LTC')
line_plot(train.close, test.close, 'Normalized Data Sequence', 'Testing Data', 'Normalized Price Data')


# In[271]:


# Price change rather than absolute price
# the first entry of each window is 0 and all other values represent the change with respect to the first value
def normalise_zero_base(df):
    # We need to normalise the column wise in order to reflect changes done with respect to the first entry
    return df / df.iloc[0] - 1

def extract_window_data(df, window=15, zero_base=True):
    #alternatively create a sliding window option
    window_data = []
    for idx in range(len(df) - window):
        tmp = df[idx: (idx + window)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

# Data prep for LSTM package
# Window = 60 points = 60 minutes
def prepare_data(df, window=15, zero_base=True, test_size=0.2):

    # train test split
    train_data, test_data = train_test_split(df, test_size)
    
    # extract window data
    X_train = extract_window_data(train_data, window, zero_base)
    X_test = extract_window_data(test_data, window, zero_base)
    
    # extract targets
    y_train = train_data.close[window:].values
    y_test = test_data.close[window:].values
    if zero_base:
        y_train = y_train / train_data.close[:-window].values - 1
        y_test = y_test / test_data.close[:-window].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test
train, test, X_train, X_test, y_train, y_test = prepare_data(df)


# In[272]:


def build_lstm_model(input_data, output_size, neurons=30,
                     activ_func='linear', dropout=0.25,
                     loss='mae', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(
              input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])
    return model
model = build_lstm_model(X_train, output_size=1)
history = model.fit(X_train, y_train, epochs=12, batch_size=15, validation_data = (X_test, y_test))


# In[265]:





# In[283]:


window = 15
targets = test[target_col][window:]
preds = model.predict(X_test).squeeze()
# convert change predictions back to actual price
preds = test.close.values[:-window] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
# The last 100 values
n = 600
line_plot (targets[-n:], preds[-n:], 'Real Price', 'Prediction', 'Real Price against the Predicted Price [Test Set]')


# In[248]:


actual_returns = targets.pct_change()[1:]
predicted_returns = preds.pct_change()[1:]
print(actual_returns)


# In[221]:


line_plot(actual_returns, predicted_returns, "Actual Returns", "Predicted Returns")


# ## Training Loss

# In[290]:


score = model.evaluate(X_test, y_test)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

print("%s: %.2f%%" % (model.metrics_names[0], score[1]*100))


# In[287]:


fig, ax = plt.subplots(1, figsize=(10, 6), dpi = 100)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss', fontsize = 16)
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize = 14)
plt.legend(['Train', 'Test'], loc='best', fontsize = 14)
plt.show()

