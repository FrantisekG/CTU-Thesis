# IMPORT LIBRARIES

import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns

########### DATA LOAD ##########

Path = "~/BTC-USD.csv"
BTC= pd.read_csv(Path,names=['Time', 'Low', 'High', 'Open', 'Close', 'Volume'])
target_col = "Close"

########### DATA SPLIT TRAIN/TEST ###############

def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df)) 
    train_data = df.iloc[:split_row] 
    test_data = df.iloc[split_row:] 
    return train_data, test_data

def line_plot(line1, line2, label1=None, label2=None, title=''):
    graf = fig, ax = plt.subplots(1, figsize=(10, 6), dpi = 100)
    ax.grid(True)
    ax.plot(line1, label=label1, linewidth=2)
    ax.plot(line2, label=label2, linewidth=2)
    ax.set_ylabel('Closing price of currency pair LTC/USD', fontsize=12)
    ax.set_xlabel('Time [120 Minutes]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=14)

train, test = train_test_split(BTC, test_size=0.2)
line_plot(train.Close, test.Close, 'Normalized Data Sequence', 'Testing Data', 'Normalized Price Data')

################ MODEL CONSTRUCTION + INDICATORS ###############

# Normalizing data

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

# Min/Max

def normalise_min_max(df):

    return (df - df.min()) / (data.max() - df.min())

# Sliding window

def extract_window_data(df, window=15, zero_base=True):
    window_data = []
    for idx in range(len(df) - window):
        tmp = df[idx: (idx + window)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

# New dataframe

def prepare_data(df, window=15, zero_base=True, test_size=0.2):

    # Train/Test split
    train_data, test_data = train_test_split(df, test_size)
    
    # Applying Sliding window
    X_train = extract_window_data(train_data, window, zero_base)
    X_test = extract_window_data(test_data, window, zero_base)
    
    # Creating label targets
    y_train = train_data.Close[window:].values
    y_test = test_data.Close[window:].values
    if zero_base:
        y_train = y_train / train_data.Close[:-window].values - 1
        y_test = y_test / test_data.Close[:-window].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test
train, test, X_train, X_test, y_train, y_test = prepare_data(BTC)

# Adding features

close = df["Close"].values
high = df["High"].values
low = df["Low"].values

# Moving Average
df['3day MA'] = df['Close'].shift(1).rolling(window = 3).mean()
df['5day MA'] = df['Close'].shift(1).rolling(window = 5).mean()
df['10day MA'] = df['Close'].shift(1).rolling(window = 10).mean()
# Standard deviation
df['STD']= df['Close'].rolling(5).std()
# Relative Strength index
df['RSI'] = RSI(df['Close'].values, timeperiod = 9)
# Bollinger Bands
up, mid, low = BBANDS(close, timeperiod = 10, nbdevup = 10, nbdevdn = 10, matype = 5)
# Stochastic 
slowk, slowd = STOCH(high, low, close, fastk_period=5, slowk_period=10, slowk_matype=0, slowd_period=10, slowd_matype=0)

###### MODEL BUILD #####

def build_lstm_model(input_data, output_size, neurons=64, activ_func='linear', dropout=0.25, loss='mae', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2]))) # One input, two output
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer, metrics=["acc"])
    return model

model = build_lstm_model(X_train, output_size=1)
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data = (X_test, y_test))

################ METRIC OF PREDICTION ###############

# Imported libraries

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# MAE
targets = test[target_col][window:]
preds = model.predict(X_test).squeeze()
mean_absolute_error(preds, y_test)
# MSE
targets = test[target_col][window:]
preds = model.predict(X_test).squeeze()
mean_squared_error(preds, y_test)
# Test loss metric
score = model.evaluate(X_test, y_test)
print('Test loss:', score[0])
print("%s: %.2f%%" % (model.metrics_names[0], score[1]*100))
# Train loss metric
score = model.evaluate(X_train, y_train)
print('Train loss:', score[0])
print("%s: %.2f%%" % (model.metrics_names[0], score[1]*100))

############### PLOTTING ###############

preds = test.Close.values[:-window] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)
# Timestep: 600 minute plot
n = 600
line_plot(targets[-n:], preds[-n:], 'Real Price', 'Prediction', 'Real Price against the Predicted Price [Test Set]')
# Timestep: 120 minute plot
n = 120
line_plot(targets[-n:], preds[-n:], 'Real Price', 'Prediction', 'Real Price against the Predicted Price [Test Set]')

# Model loss plot

fig, ax = plt.subplots(1, figsize=(10, 6), dpi = 100)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss', fontsize = 16)
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize = 14)
plt.legend(['Train', 'Test'], loc='best', fontsize = 14)
plt.show()


