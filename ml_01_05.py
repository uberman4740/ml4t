
# coding: utf-8

# In[2]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time

from ml_utilityfunc import *

'''Ipython notebook code for Machine Learning for Trading, L0105'''

# Incomplete Data:
    # Even though financial data is well recorded, it can be faulty and have
    # missing information
    # reality: - data is an amalgamation of information of many different sources
    #          - not all stocks trade every day (IPOs, privatizations, buyouts, acquisitions)
    # What do we do when there are a lot of NaN's in the data?
        # - Should we interpolate? there was no trading in middle of missing data...
        # - Instead we can fill forward from the last previous known value, which
        #   is a more realistic representation of the behavior in the data
        # - This is because we only want to be able to do real-time processing, 
        #   and not peak into the future!
        # - Fill backwards for missing data second
        # - in pandas, use fillna() function

def fill_missing_values(df):
    pass
    df.fillna(method='ffill',inplace='TRUE')
    df.fillna(method='bfill',inplace='TRUE')        
        
def test_run():
    
    dates = pd.date_range('2005-12-31','2012-12-07')
    symbols = ['JAVA','FAKE1','FAKE2']
    df = get_data(symbols, dates)
    #df = df['FAKE2']
    #df.fillna(method='ffill',inplace='TRUE')
    #df.fillna(method='bfill',inplace='TRUE')
    fill_missing_values(df)
    ax = df.plot(title="Missing Data Example", label='FAKE2')
    
    
    #fillna(method = 'ffill')


# In[3]:

test_run()


# In[ ]:



