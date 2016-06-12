
# coding: utf-8

# In[9]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time

from ml_utilityfunc import *

# Portfolios: 
    # Now move toward generating statistics on portfolios
    # We assume the allocations of assets in porfolio sum to one
    # Assumptions:
        # start_val = 1,000,000
        # start_date = 01-01-2009
        # end_date = 12-31-2011
        # symbols = ['SPY','XOM','GOOG','GLD']
        # allocs = [0.4, 0.4, 0.1, 0.1] --> beginning allocations
    # Calculating total value of portfolio day by day:
        # normalize prices of each stock in portfolio
            # normed = prices/prices[0]
        # allocated normed values across portfolio
            # alloced = normed*(allocs)
        # include initial value of portfolio weighted by alloced 
            # value of position each day
            # pos_vals = alloced * start_val
        # sum across each day to get total value of portfolio each day
            # port_val = pos_vals.sum(axis=1) --> sum across rows

# Portfolio Statistics:
    # given port_val --> day-by-day portfolio value
    # What to calculate:
        # daily_rets (exclude first 0 w/ daily_rets=daily_rets[1:])
        # cum_ret --> how much value of portfolio has gone up from beginning to end
            # cum_ret = (port_val[-1]/port_val[0])-1
        # avg_daily_ret --> average daily returns during the time period
            # avg_daily_ret = daily_rets.mean()
        # std_daily_ret --> variation from the mean of the daily returns during the time period
            # std_daily_ret = daily_rets.std()
        # sharpe_ratio --> risk-adjusted return; measure of return on risk of the portfolio
            # considers return relative to the variation/volatility/std of the daily returns
            # all else equal:
                # lower risk is better
                # higher return is better
            # considers risk-free rate of return to consider portfolio relative to safe asset alternative
                # Fed has driven interest rates very low so rfr ~= 0
            # 3 variables for use:
                # Rp = portfolio return
                # Rf = risk free rate of return
                # sp = std dev of portfolio return
            # S = E[Rp-Rf]/std[Rp-Rf] --> ex ante measure
                # higher volatility, lower sharpe
                # higher returns, higher sharpe
                # higher risk free rate, lower sharpe
            # sharpe_ratio = mean(daily_rets - daily_rfr) / std(daily_rets - daily_rfr)
            # to simplify equation, can convert the annual amount using a shortcut:
                # using annual rfr --> daily_rfr = (1.0 + rfr_annual)^(1/252) - 1
                # we approximate daily_rfr = 0, though
                # constant daily_rfr will result in:
                    # sharpe_ratio = mean(daily_rets - daily_rfr) / std(daily_rets)
            # Note: - Sharpe ratio can vary widely depending on how frequently you sample
            #       - e.g. daily vs. monthly vs. annually
            #       - sharpe ratio originally designed as an annualized measure
            #       - to sample other than annually, need multiply by adjustment factor K
            #       - that is, SR_samp = K*SR where K = sqrt(n_samples per year)
            #       - therefore daily then K is based on:
            #            --> 252 samples, weekly then 52, monthly then 12
            # daily sharpe ratio:
                # daily_SR = sqrt(252)*(mean(daily_rets - daily_rfr) / std(daily_rets))

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
    
def test_run():
    # initialize dates
    start_date = '2009-01-01'
    end_date = '12-31-2011'
    dates = pd.date_range(start_date,end_date)
    
    # generate dataframe of portfolio inputs
    start_val = 1000000
    symbols = ['SPY','XOM','GOOG','GLD']
    allocs = [0.4, 0.4, 0.1, 0.1]
    df = get_data(symbols, dates, dropna=True)
    
    # generate portfolio
    #normed = normalize_data(df)
    #alloced = normed*allocs
    #pos_vals = alloced*start_val
    #port_val = pos_vals.sum(axis=1)
    # print port_val
    port_val = get_portfolio_value(df,allocs,start_val)
    
    # daily return
    #daily_ret_p = (port_val/port_val.shift(1))-1
    #daily_ret_p.ix[0] = 0
    #daily_ret_p = daily_ret_p[1:]
    # print daily_ret_p
    daily_ret_p = get_portfolio_daily_ret(port_val)
    
    # generate portfolio statistics
    cum_ret_p = (port_val[-1]/port_val[0])-1  # cumulative return
    std_daily_ret_p = daily_ret_p.std() # daily return standard deviation
    avg_daily_ret_p = daily_ret_p.mean() # average daily return
    # SR_p = np.sqrt(252)*(avg_daily_ret_p)/(std_daily_ret_p)
    
    #daily_ret = compute_daily_returns(df)
    #SR = np.sqrt(252)*daily_ret.mean()/daily_ret.std()
    #print SR
    
    SR_p = get_sharpe_ratio(daily_ret_p, daily_ret_p.std(), annual_rfr=0, mod=1)
    print "Sharpe Ratio = ", SR_p


# In[10]:

test_run()


# In[ ]:



