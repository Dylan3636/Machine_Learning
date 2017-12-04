# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:55:42 2017

@author: dylan
"""
import numpy as np
from gaussians import *

def get_posterior_paramters(X, y, sig_y=0.1, w_0=None, var_w=[1]):
   var_y = sig_y**2
   n, d = np.shape(X)
   d+=1
   phi = np.concatenate([X, np.ones([n, 1])], axis=1)
   if w_0 is None:
       w_0 = np.zeros(d)
   var_w = np.diag(var_w)
   
   
   V_0 = var_w*np.eye(d)
   Inv = np.linalg.solve(V_0, var_y*np.eye(d))
   V_n = np.linalg.solve( Inv + np.dot(phi.T, phi), var_y*np.eye(d))
   
   Inv_V = np.dot(V_n, np.linalg.solve(V_0, np.eye(d) ))
   w_n = np.dot(np.dot(V_n,Inv_V), w_0) + (1/var_y)*np.dot(np.dot(V_n, phi.T), y)
   
   return w_n, V_n

def get_prediction_parameters(x, sig_y, w_n, V_n):
    var_y = sig_y**2
    mew = np.dot(x, w_n);
    cov = np.dot(np.dot(x.T, V_n),x) + var_y;
    return mew, cov

def calculate_log_likelihood(X, y, sig_y, w_n, V_n):
    var_y = sig_y**2
    n = np.shape(X)[0]
    log_likelihood=0
    for i in range(n):
        x = X[i, :]
        mew, cov = get_prediction_parameters(x,var_y, w_n, V_n)
        log_likelihood += log_gaussian_pdf(y[i], np.dot(x, mew), cov)
    return log_likelihood
