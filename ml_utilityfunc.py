
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

def symbol_to_path(symbol,base_dir="data"):
    # return filepath of csv corresponding to a given ticker symbol
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols,dates,dropna=False):
    # function to read in stock data (adj. close) for given symbols
    df = pd.DataFrame(index=dates)
    
    # add SPY for reference if absent
    if 'SPY' not in symbols:
        symbols.insert(0,'SPY')
        
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date','Adj Close'],
                              na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        if dropna == True:
            df = df.join(df_temp,how='inner')
        else:
             df = df.join(df_temp)
        
    return df.sort_index(ascending=True) # return list sorted by date

def plot_data(df, title="Stock Prices",ylabel="Price",xlabel="Date"):
    ax = df.plot(title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    
def plot_selected(df, columns, start_index, end_index):
    df_sliced = df.ix[start_index:end_index,columns]
    plot_data(df_sliced)

def normalize_data(df):
    return df/df.ix[0,:]

def fill_missing_values(df):
    pass
    df.fillna(method='ffill',inplace='TRUE')
    df.fillna(method='bfill',inplace='TRUE')  


# In[ ]:



