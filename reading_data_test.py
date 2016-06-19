
# coding: utf-8

# In[5]:

import pandas_datareader.data as web
import datetime
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2016, 6, 19)

google = web.DataReader("GOOG", 'yahoo', start, end)

google['Adj Close']


# In[ ]:




# In[ ]:



