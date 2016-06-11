
# coding: utf-8

# In[26]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time

from ml_utilityfunc import *

'''Ipython notebook code for Machine Learning for Trading, L0106'''

# Histograms and Scatterplots
    # It is more useful to consider daily returns relative to different stocks
    # Generally, distributions of daily returns follow an approximate normal distribution
    # These histograms will be skewed towards positive or negative returns
        # As a result, you want different statistics on the distribution of returns
            # mean     --> tells us about the central tendency in the data
            # std      --> tells us about the variation from the mean
            # kurtosis --> tells us how different the distribution is from the traditional dist
            #              which gives us information about the tails in the dist
            #              - positive kurtosis: fat tails
            #              - negative kurtosis: skinny tails
    # Linear regression:
        # Slope (beta) --> relationship between returns of two stocks
        # Intercept (alpha) --> how much on average stock on horiz axis performs relative to vertical
        # Scatter of stock vs. SPY gives measure of performance relative to the market
        # Note: - slope does not imply correlation necessarily
        #       - correlation gives you a measure of how tightly data fits the regressed line
        # For linereg, use numpy polyfit function
        
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

def hist_one_stock():
    dates = pd.date_range('2009-01-01','2012-12-31')
    symbols = ['SPY']
    df = get_data(symbols, dates, dropna=True)
    plot_data(df)
    
    daily_ret = compute_daily_returns(df)
    plot_data(daily_ret, title="Daily Returns", 
              ylabel="Daily Returns", xlabel="Date")
    
    # generate histogram
    daily_ret.hist(bins=40)
    
    # generate stats
    mean = daily_ret['SPY'].mean()
    std = daily_ret['SPY'].std()
    kurt = daily_ret['SPY'].kurtosis()
    print "mean = ", mean
    print "\nstd = ", std
    print "\nkurtosis = ", kurt
    
    # plot stats on hist
    plt.axvline(mean, color='r',linestyle='dashed',linewidth=2)
    plt.axvline(mean-std, color='y',linestyle='dashed',linewidth=2)
    plt.axvline(mean+std, color='y',linestyle='dashed',linewidth=2)
    plt.show()

def hist_two_stocks():
    dates = pd.date_range('2009-01-01','2012-12-31')
    symbols = ['SPY','XOM']
    df = get_data(symbols, dates, dropna=True)
    plot_data(df)
    
    daily_ret = compute_daily_returns(df)
    plot_data(daily_ret, title="Daily Returns", 
              ylabel="Daily Returns", xlabel="Date")
    
    # generate histograms
    daily_ret['SPY'].hist(bins=40,label='SPY')
    daily_ret['XOM'].hist(bins=40,label='XOM')
    plt.legend(loc='upper right')
    #plt.show()
    
    # generate stats
    mean = daily_ret.mean()
    std = daily_ret.std()
    kurt = daily_ret.kurtosis()
    print "mean = ", mean
    print "\nstd = ", std
    print "\nkurtosis = ", kurt
    
    # plot stats on hist
    plt.axvline(mean[0], color='r',linestyle='dashed',linewidth=2)
    plt.axvline(mean[0]-std[0], color='y',linestyle='dashed',linewidth=2)
    plt.axvline(mean[0]+std[0], color='y',linestyle='dashed',linewidth=2)
    
    plt.axvline(mean[1], color='g',linestyle='dashed',linewidth=2)
    plt.axvline(mean[1]-std[1], color='c',linestyle='dashed',linewidth=2)
    plt.axvline(mean[1]+std[1], color='c',linestyle='dashed',linewidth=2)
    plt.show()

def scatter_two_stocks():
    dates = pd.date_range('2009-01-01','2012-12-31')
    symbols = ['SPY','XOM','GLD']
    df = get_data(symbols, dates, dropna=True)
    plot_data(df)
    
    daily_ret = compute_daily_returns(df)
    plot_data(daily_ret, title="Daily Returns", 
              ylabel="Daily Returns", xlabel="Date")
    
    # generate histograms (SPY vs. XOM)
    daily_ret.plot(kind='scatter',x='SPY',y='XOM')
    beta_XOM,alpha_XOM = np.polyfit(daily_ret['SPY'],daily_ret['XOM'],1)
    linreg_XOM = alpha_XOM + daily_ret['SPY']*beta_XOM
    plt.plot(daily_ret['SPY'],linreg_XOM,'-',color='r')
    plt.show()
    
    # generate histograms (SPY vs. GLD)
    daily_ret.plot(kind='scatter',x='SPY',y='GLD')
    beta_GLD,alpha_GLD = np.polyfit(daily_ret['SPY'],daily_ret['GLD'],1)
    linreg_GLD = alpha_GLD + daily_ret['SPY']*beta_GLD
    plt.plot(daily_ret['SPY'],linreg_GLD,'-',color='r')
    plt.show()
    
    # generate correlation
    corr = daily_ret.corr(method='pearson')
    
    # print results 
    print "beta_XOM = ", beta_XOM
    print "alpha_XOM = ", alpha_XOM
    
    print "\nbeta_GLD = ", beta_GLD
    print "alpha_GLD = ", alpha_GLD
    
    print "\n",corr
    


# In[5]:

hist_one_stock()


# In[6]:

hist_two_stocks()


# In[ ]:




# In[27]:

scatter_two_stocks()


# In[ ]:



