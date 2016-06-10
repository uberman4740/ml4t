
# coding: utf-8

# In[16]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# problems to solve: 
    # want to read particular date ranges --> index by dates
    # reading in multiple stocks
    # align dates between separate stocks
    # proper date order (reverse list)
    # remember 252 trading days

# building the dataframe we want:
    # (df1) start with empty dataframe with all dates we are interested in
    # --> target dataframe includes the date range we are interested in
    # want to load df1 with info from each stock
    # --> separately read in a dataframe for each stock (e.g. dfSPY dfIBM dfAAPL dfGLD)
    # then you need to join the dataframes (using adjoin function)
        # match dates from target dataframe to df1 (date range frame)
        # eliminate rows with dates missing stock data
    # repeat for each additional target stock symbol

# you can slice a dataframe indexed by dates using datetime objects
# this looks liks df[start_date:end_date,['column1','column2']]
    # row slicing:
    # column slicing:

# it helps for visualization to start all data at some index point, 
# such as the price of SPY at time 0 
    # --> then you compare to 1 vs. comparing btwn different prices
    # that is, you want to NORMALIZE price data
    
def create_empty_df():
    # create datetime array with trailing zeros for time
    start_date='2010-01-22'
    end_date='2010-01-26'
    dates=pd.date_range(start_date,end_date)
    #print dates[0]
    # create dataframe indexed by dates
    df1 = pd.DataFrame(index=dates)
    #print df1
    
    # read SPY data into temporary dataframe
    #dfSPY = pd.read_csv("data/SPY.csv", index_col='Date',
    #                    parse_dates=True, usecols=['Date','Adj Close'],
    #                    na_values=['nan'])

    # join the two dataframes with DataFrame.join function
    # df1 = df1.join(dfSPY)
    #print df1
    
    # drop rows where adj close in NaN
    # df1 = df1.dropna()
    # print df1
    
    # can do join and dropna in one step
    # df1 = df1.join(dfSPY,how='inner')
    # print df1
    
    symbols = ['SPY','GOOG','IBM','GLD']
    for symbol in symbols:
        df_temp = pd.read_csv("data/{}.csv".format(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date','Adj Close'],
                              na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df1 = df1.join(df_temp,how='inner')
    print df1

def symbol_to_path(symbol,base_dir="data"):
    # return filepath of csv corresponding to a given ticker symbol
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols,dates):
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
        df = df.join(df_temp,how='inner')
        
    return df.sort_index(ascending=True) # return list sorted by date

def plot_data(df, title="Stock Prices"):
    ax = df.plot(title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()
    
def plot_selected(df, columns, start_index, end_index):
    df_sliced = df.ix[start_index:end_index,columns]
    plot_data(df_sliced)

def normalize_data(df):
    return df/df.ix[0,:]


# In[17]:

dates=pd.date_range('2010-01-01','2010-12-31')
symbols=['GOOG','IBM','GLD']

df1 = get_data(symbols,dates)


# row slicing using DataFrame.ix[] --> month of January
df_jan = df1.ix['2010-01-01':'2010-01-31']

# column slicing
df_GOOG = df1['GOOG']
df_IBM_GOOG = df1[['IBM','GOOG']]

# row-col slicing using .ix selector
df_SPY_IBM_jan = df1.ix['2010-01-01':'2012-01-31',['SPY','IBM']]

# plot data
plot_data(df_SPY_IBM_jan_norm)
plot_selected(df1,['IBM','GOOG'],'2010-03-01','2010-03-31')

# normalize price data
plot_data(normalize_data(df1))



# In[ ]:




# In[ ]:



