import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from math import floor
from math import sqrt
from scipy import stats
import os
# visualize 
import matplotlib.pyplot as plt
import seaborn as sns
# working with dates
from datetime import datetime
# to evaluated performance using rmse
from sklearn.metrics import mean_squared_error
from math import sqrt 
from sklearn.model_selection import train_test_split
# for tsa 
import statsmodels.api as sm

def get_data():
    filename = 'ETH-USD.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read .csv file from github
        df = pd.read('https://raw.githubusercontent.com/albertopuentes/Individual-Project/master/ETH-USD.csv')

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_file(filename)

        # Return the dataframe to the calling code
        return df  

def clean_data(eth):
    # handle null data by interpolating missing values 
    eth = eth.interpolate(method='linear')
    # Convert Date to DateTime format and set as index
    eth.Date = pd.to_datetime(eth.Date)
    eth = eth.set_index('Date').sort_index()
    # drop 'Adj Close'
    eth = eth.drop(columns=('Adj Close'))
    # drop columns that aren't going to be used as features or in feature engineering
    eth = eth.drop(columns=['Open', 'High', 'Low'])
    return eth

def delta_rolling(eth):
    '''function will take in cleaned data and add % change day-over-day & rolling avg columns'''
    # Create % change Day-over-Day data columns
    eth['close_DoD'] = round(eth.Close.pct_change(), 2)
    eth['vol_DoD'] = round(eth.Volume.pct_change(), 2)
    # Create Close 50 & 200 day rolling averages (Close & Volume)
    dfm = eth.assign(
    rolling_50C=lambda eth: eth.Close.rolling(50).mean(),
    rolling_200C=lambda eth: eth.Close.rolling(200).mean(),
    rolling_50V=lambda eth: eth.Volume.rolling(50).mean(),
    rolling_200V=lambda eth: eth.Volume.rolling(200).mean())
    return dfm

### CREATE NEW INDICATOR FEATURE FUNCTIONS ###

# Exponential Moving Average Indicator Function
def EMA(df, period, column):
    return df[column].ewm(span=period, adjust=False).mean()

# Relative Strength Indicator Function (Exponential Moving Average)
def RSI(df, period):
    window_length= period
    # Dates
    start = df.index.min()
    end = df.index.max()
    # Get the difference in price from previous step
    delta = df.Close.diff()
    # Get rid of the first row, which is NaN since it did not have a previous 
    # row to calculate the differences
    delta = delta[1:]
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.clip(lower=0), delta.clip(upper=0)
    # Calculate the EWMA
    roll_up1 = up.ewm(span=window_length).mean()
    roll_down1 = down.abs().ewm(span=window_length).mean()

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI = 100.0 - (100.0 / (1.0 + RS1))
    
    df['RSI'+str(period)] = RSI
    return df

### UTILIZE FUNCTIONS TO ADD FEATURES TO DATA ###

def indicators(df):
    '''Run RSI & EMA functions to create columns in dfm'''
    RSI(df, 7)
    RSI(df, 12)
    RSI(df, 26)
    df['EMA7'] = EMA(df, 7, 'Close')
    df['EMA12'] = EMA(df, 12, 'Close')
    df['EMA26'] = EMA(df, 26, 'Close')
    # Add Crossover indicator (rollling 50 vs rolling 200)
    df['momentum-cross'] = round(df.rolling_50C/df.rolling_200C, 2)
    return df

# Add On Balance Volume indicator
def obv(df):  
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV'] = round(obv, 0)
    return df

# Calculate Moving Average Convergence/Divergence
def macd(df):
    exp1 = df.Close.ewm(span=12, adjust=False).mean()
    exp2 = df.Close.ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    return df

# Calculate and add Target +/- column
def target(df):
    df['up_down'] = np.where(df.Close.shift(-1) > df.Close, 1, 0) 
    return df