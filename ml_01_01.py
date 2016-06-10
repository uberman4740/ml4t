
# coding: utf-8

# In[12]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def get_max_close(symbol):
    df = pd.read_csv("data/{}.csv".format(symbol)) # read in data corresponding to symbol
    return df['Close'].max() # compute and return max

def get_mean_volume(symbol):
    df = pd.read_csv("data/{}.csv".format(symbol))
    return df['Volume'].mean()

def test_run_indexing():
    df = pd.read_csv("data/AAPL.csv")
    # print df.head() # print entire dataframe
    # note: df.head() --> first 5 values
    #       df.tail() --> last 5 values
    #       first n rows then .tail(n)
    print df[10:21] # print rows indexed btwn 10 and 21 (non-inclusive --> 10 to 20)
    
def test_print_stats():
    for symbol in ['AAPL','IBM']:
        print "Max Close"
        print symbol, get_max_close(symbol)
        print symbol, get_mean_volume(symbol)

def test_plot_high_aapl():
    df = pd.read_csv("data/AAPL.csv")
    #print df['Adj Close']
    df[['Close','Adj Close']].plot()
    plt.show()

def test_plot_high_ibm():
    df = pd.read_csv("data/IBM.csv")
    #print df['High']
    df['High'].plot()
    plt.show()

    


# In[13]:

test_plot_high_aapl()


# In[ ]:



