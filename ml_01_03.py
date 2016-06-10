
# coding: utf-8

# In[76]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time

from ml_utilityfunc import *

# Numpy is a python wrapper around C/Fortran and uses ndarrays 
# that make the syntax very similar to MATLAB. Numpy allows you to
# do all kinds of arithmetic/matrix/numerical operations. 

# Pandas is essentially a wrapper for numpy using ndarrays 
    # --> the data in the df is an ndarray
    # as a result, you can use a dataframe as an ndarray directly
    
# ndarray syntax
    # nd[row,column] --> indexing
    # columns and rows begin at zero
    # slicing:
        # nd[row_start:row_end,col_star:col_end] --> just like MATLAB
        # note that the end is not inclusive!
        # --> using colon by itself means we get all the rows/cols before
        #     or after the colon
    # last row, all columns --> nd[-1,:]
    # last column, all rows --> nd[:,-1]
    # note that you can use another array of indices to index an array
    
# numpy masking: 
    # numpy masking is a useful way of easily finding values above or 
    # below some values (e.g. the mean)
    
def test_run():
    
    # t1 = time.time()
    
    # print np.array([(2,3,4),(5,6,7)])
    
    # prefilled arrays
    #print np.empty((4,5)) # empty array 
    #print np.ones((4,5))
    #print np.ones((4,5),dtype=np.int8)
    
    # generate random numbers sampled uniformly between 0.0 and 1.0
    #print np.random.random((4,5))
    #print np.random.rand(4,5)
    
    # generate random numbers sampled normally between 0.0 and 1.0
    #print np.random.normal(size=(4,5)) # mean=0, sd=1
    #print np.random.normal(50,10,size=(4,5)) # mean=50, sd=10
    
    # random integers
    #print np.random.randint(10) # single int btwn 0 and 10
    #print np.random.randint(0,10) # single int explicitly btwn 0 and 10
    #print np.random.randint(0,10,size=5) # 1x5 array of random ints btwn 0 and 10
    #print np.random.randint(0,10,size=(2,3)) # 2x3 array of random ints btwn 0 and 10
    
    # accessing shape of the given array
    #a = np.random.random((5,4))
    #print a
    #print a.shape
    #print a.shape[0] # number of rows
    #print a.shape[1] # number of columns
    #print len(a.shape) # number of dimensions 
    #print a.size # total number of elements in array
    #print a.dtype # datatype of array elements
    
    # operations on ndarrays
    np.random.seed(693)
    a = np.random.randint(0,10,size=(5,4))
    print "Array:\n", a
    print "Sum of all elements: ", a.sum()
    
    # can also sum on row or column
    print "Sum along rows: ", a.sum(axis=0) 
    print "Sum along columns: ", a.sum(axis=1)
    
    # finding minimum/maximum
    print "Minimum of each column: ", a.min(axis=0) # min across all rows in each col
    print "Maximum of each row: ", a.max(axis=1) # max across all cols in each row
    print "Mean of all elements: ", a.mean()
    
    # t2 = time.time()
    # print "The time taken by above is ", t2-t1, " seconds"
    
    
    

def test_run_2():
    nd1 = np.random.random((1000,10000))
    
    res_manual,t_manual = time_exec(manual_mean,nd1)
    res_np,t_np = time_exec(numpy_mean,nd1)
    print "Manual: {:.6f} ({:.3f} secs) vs. Numpy: {:.6f} ({:.3f} secs)".format(res_manual,t_manual,res_np,t_np)
    
    assert abs(res_manual - res_np) <= 10e-6, "Results aren't equal!"
    
    speedup = t_manual/t_np
    print "Numpy mean is ", speedup, "times faster than manual for loops."
    
def get_max_index(a):
    #max_a = a.max()
    #for i in range(0,len(a)):
    #    if a[i] == max_a:
    #        return
    return a.argmax() # does all of above in one step

def time_exec(func, *args):
    t0 = time.time()
    result = func(*args)
    t1 = time.time()
    return result, t1-t0

def manual_mean(arr):
    cum_sum = 0
    for i in xrange(0,arr.shape[0]):
        for j in xrange(0,arr.shape[1]):
            cum_sum = cum_sum + arr[i,j]
    return cum_sum/arr.size

def numpy_mean(arr):
    return arr.mean()


# In[77]:

test_run_2()

