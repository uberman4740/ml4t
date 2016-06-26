
# coding: utf-8

# ## k-Nearest-Neighbor Algorithm 
# - developed a kNN algorithm for simple prediction of the stock price from a single feature input 
# 

# In[15]:

import scipy as sp
import numpy as np
import matplotlib
import math

from ml_utilityfunc import *

class kNNLearner:
    def __init__(self,k): 
        self.k = k
        
    def train(self,X,Y):
        # fit a line to the data
        # find an m and a b --> parameters of linear model
        # self.m, self.b, intercept,rvalue,pvalue,stderr = sp.stats.linregress(X,Y) # use algo you want from SciPy and Numpy
        self.xt = X
        self.yt = Y
        
    def query(self,X):
        y_pred = []
        for Xt in X:
            kNN_indx = np.argsort(get_distance(Xt,self.xt))[:self.k]
            y_pred.append(np.average(self.yt[kNN_indx]))
        return y_pred


# In[36]:

def test_run():
    # initialize training data
    start_datet = '2011-01-01'
    end_datet = '12-31-2013'
    datest = pd.date_range(start_datet,end_datet)
    symbols = ['SPY','GOOG']
    df_train = get_data(symbols, datest, dropna=True)
    
    pred_win = 5
    learner = kNNLearner(k=pred_win)
    learner.train(df_train['GOOG'][:-pred_win],df_train['SPY'][pred_win:])
    
    start_date = '2014-01-01'
    end_date = '12-31-2014'
    
    dates = pd.date_range(start_datet,end_datet)
    df_test = get_data(symbols, dates, dropna=True)
    
    ypredict = learner.query(df_test['GOOG'][:-pred_win])
    
    plt.plot(ypredict,df_test['SPY'][pred_win:],'k.')
    plt.xlabel('SPY Predicted Price')
    plt.ylabel('SPY Actual Price')
    plt.title('SPY kNN Learner-Predictor')
    print np.corrcoef(ypredict,df_test['SPY'][pred_win:])
    
    # print learner.query([704.25])


# In[37]:

test_run()


# In[ ]:



