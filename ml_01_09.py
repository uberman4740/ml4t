
# coding: utf-8

# In[31]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import scipy.optimize as spo
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import os
import time

from ml_utilityfunc import *

# Portfolio optimizer:
    # Definition:
        # Given a set of assets and a time period, pick the set of allocations
        # that maximize performance
    # Performance:
        # could be a number of metrics; cumulative return, volatility, and
        # risk adjusted return (use Sharpe ratio)
    # Framing:
        # 1. objective function f(x) = -(Sharpe Ratio)
        # 2. initial guess (x = allocations)
        # 3. call the optimizer
    # Ranges & Constraints:
        # limits on values for X --> only worth looking between 0 and 1, 
        #                            where sum over abs(X) equals 
        
def sharpe(allocs,df):
    start_val = 1000000
    port_val = get_portfolio_value(df,allocs,start_val)
    daily_ret_p = get_portfolio_daily_ret(port_val)
    SR = get_sharpe_ratio(daily_ret_p, daily_ret_p.std(), annual_rfr=0, mod=1)
    return -1*SR

def optimizer(df, sharpe):
    allocs0 = [0.25, 0.25, 0.25, 0.25]
    cons={'type':'eq', 'fun': con}
    min_result = spo.minimize(sharpe,allocs0,args=df,constraints=cons,
                              bounds=[(0,1),(0,1),(0,1),(0,1)],
                              method='SLSQP',options={'disp':True})
    return min_result.fun,min_result.x

def con(allocs):
    return allocs.sum()-1

def gen_df():
    # generate dataframe of portfolio inputs
    dates = pd.date_range('2010-01-01','2010-12-31')
    start_val = 1000000
    symbols = ['GOOG','AAPL','GLD','XOM']
    allocs = [0.25, 0.25, 0.25, 0.25]
    df = get_data(symbols, dates, dropna=True)
    
    max_sharpe,allocs_min = optimizer(df,sharpe)
    allocs_min = allocs_min.round(1)
    dictn = dict(zip(symbols,allocs_min))
    
    print "\nOptimal Portfolio:\n", "Sharpe Ratio = {}".format(max_sharpe)
    print "Optimal Allocations = ", dictn
        


# In[32]:

gen_df()


# In[ ]:




# In[ ]:




# In[ ]:



