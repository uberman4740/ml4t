
# coding: utf-8

# In[96]:

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

# Optimizers:
    # Algorithms that allow you to:
        # - find minima/maxima of functions, build
        # - build parameterized models based on data
        # - refine allocations to stocks in portfolios
    # Three key steps:
        # 1. provide a function to minimize/maximize
        # 2. provide an initial guess (seed input)
        # 3. call the optimizer
    # Defeating a minimizer:
        # - multiple minima (both local and global)
        # - nonlinearities/discontinuities
        # - locations without gradients
    # Convex problems:
        # Formal definition - a real valued function f(x) defined on an interval
        #                     is called convex if the line segment between any two
        #                     points on the function lies above the graph
        # Convex problems have only one local minima that is therefore the global min
        # Cannot have flat regions
        # Can also consider multi-dimensional problems
    # Building a parameterized model from data:
        # e.g. f(x) = m*x + b --> f(x) = c0 * x + c1
        # Let's say we have some data that gives us information of how x and y vary
        # Issue is reframing the problem for the minimizer 
            # - minimizing sum of squared deviations between model and data
            # - let e = error between data and model

# Simple minimizer
def f0(X):
    Y = (X-1.5)**2+0.5
    #print "X = {}, Y = {}".format(X,Y) # to trace minimization steps
    return Y

def test_run0():
    X0 = 2.0 # initial guess
    min_result = spo.minimize(f0,X0,method='SLSQP',options={'disp':True})
    print "Minima found at:"
    print "X = {}, Y = {}".format(min_result.x,min_result.fun)
    
    # plot results
    xvals = np.linspace(0,3,41)
    yvals = f0(xvals)
    plt.plot(xvals, yvals)
    plt.plot(min_result.x,min_result.fun,'ro')
    plt.axis([0, 3, 0, 3])
    plt.title('Minima of objective function Y = (X-1.5)^2+0.5')
    plt.show()

# Parameterized model builder
def error_line(line, data): # error function for linear model
    # line: tuple/list/array (c0, c1)
    # data: 2D array with (x,y) points
    err = np.sum((data[:,1]-(line[0]*data[:,0]+line[1]))**2)
    return err

def fit_line(data,error):
    line0 = [1,1]
    min_result = spo.minimize(error,line0,args=(data),method='SLSQP',options={'disp':True})
    params = min_result.x
    return params
    
def test_run1():
    l_orig = np.float32([4,2])
    Xorig = np.linspace(0,10,21)
    Yorig = l_orig[0]*Xorig + l_orig[1]
    plt.figure(0)
    plt.plot(Xorig, Yorig,'b-',linewidth=2.0,label="Original Line")
    
    # generate noisy data
    noise_sig = 3.0
    noise = np.random.normal(0,noise_sig,Yorig.shape)
    data = np.asarray([Xorig,Yorig+noise]).T
    plt.figure(0)
    plt.plot(data[:,0], data[:,1],'ro',label="Data")
    line0 = [1,1]
    plt.legend(loc="upper left")
    
    # gen parameterized model
    params = fit_line(data,error_line)
    model = params[0]*Xorig + params[1]
    plt.figure(1)
    plt.plot(Xorig,model,'g-',linewidth=2.0,label="Model")
    plt.plot(data[:,0], data[:,1],'ro',label="Data")
    plt.legend(loc="upper left")
    
# Polynomial parameterized models
def error_poly(params, data): # error function for linear model
    # line: tuple/list/array (c0, c1)
    # data: 2D array with (x,y) points
    err = np.sum((data[:,1]-(np.polyval(params,data[:,0])))**2)
    return err

def fit_poly(data,error):
    poly0 = [1,1,1,1,1]
    min_result = spo.minimize(error,poly0,args=(data),method='SLSQP',options={'disp':True})
    params = min_result.x
    return params

def test_run2():
    p_orig = np.float32([1.5,-10,-5,60,50])
    Xorig = np.linspace(0,10,21)
    Yorig = p_orig[0]*Xorig**4 + p_orig[1]*Xorig**3 + p_orig[1]*Xorig**2 + p_orig[1]*Xorig + p_orig[0]
    plt.figure(0)
    plt.plot(Xorig, Yorig,'b-',linewidth=2.0,label="Original Line")
    
    # generate noisy data
    noise_sig = 250
    noise = np.random.normal(0,noise_sig,Yorig.shape)
    data = np.asarray([Xorig,Yorig+noise]).T
    plt.figure(0)
    plt.plot(data[:,0], data[:,1],'ro',label="Data")
    line0 = [1,1]
    plt.legend(loc="upper left")
    
    params = fit_poly(data,error_poly)
    model = params[0]*Xorig**4 + params[1]*Xorig**3 + params[2]*Xorig**2 + params[3]*Xorig + params[4]
    plt.figure(1)
    plt.plot(Xorig,model,'g-',linewidth=2.0,label="Model")
    plt.plot(data[:,0], data[:,1],'ro',label="Data")
    plt.legend(loc="upper left")
    print params


# In[97]:

test_run()


# In[98]:

test_run1()


# In[99]:

test_run2()

