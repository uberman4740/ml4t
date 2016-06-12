
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

'''Ipython notebook code for Machine Learning for Trading, Utility Functions'''

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
    
def compute_daily_returns(df):
    # daily_ret[t]=(price[t]/price[t-1])-1
    #daily_ret = df.copy()
    #daily_ret[1:] = (df[1:]/df[:-1].values)-1
    daily_ret = (df/df.shift(1))-1
    daily_ret.ix[0,:] = 0
    return daily_ret

def compute_cumulative_returns(df):
    daily_ret = compute_daily_returns(df)
    return daily_ret.cumsum()

def get_bollinger_bands(rm,rstd):
    upper_band = rm + 2*rstd
    lower_band = rm - 2*rstd
    return upper_band, lower_band

def get_portfolio_value(df,allocs,start_val):
    normed = normalize_data(df)
    alloced = normed*allocs
    pos_vals = alloced*start_val
    port_val = pos_vals.sum(axis=1)
    return port_val
                
def get_portfolio_daily_ret(port_val):
    daily_ret_p = (port_val/port_val.shift(1))-1
    daily_ret_p.ix[0] = 0
    daily_ret_p = daily_ret_p[1:]
    return daily_ret_p

def get_sharpe_ratio(returns, std_returns, annual_rfr=0, mod=0):
    if mod == 0:
        K = 1
        rfr = annual_rfr
    elif mod == 1: # daily mode
        K = np.sqrt(252)
        rfr = np.power((1.0 + annual_rfr),(1/252)) - 1
    elif mod == 2: # weekly mode
        K = np.sqrt(52)
        rfr = np.power((1.0 + annual_rfr),(1/52)) - 1
    elif mod == 3: # monthly mode
        K = np.sqrt(12)
        rfr = np.power((1.0 + annual_rfr),(1/12)) - 1
    adj_returns = returns - rfr
    return K*(adj_returns.mean()/std_returns)


# In[ ]:



