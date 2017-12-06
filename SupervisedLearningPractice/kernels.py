# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 00:13:49 2017

@author: dylan
"""
import numpy as np
class rbf_kernel:
    def __init__(self, sig_f, l):
        self.var_f = sig_f**2
        self.l = l
    
    def kernel(self, X, X_prime):
        X=np.matrix(X)
        X_prime = np.matrix(X_prime)
        N = np.size(X,0)
        M = np.size(X_prime, 0)
        K = np.zeros([N, M])
        try:
            np.size(X,1)
        except:
            X=np.reshape(X,[N,1])
        try:
            np.size(X_prime,1)
        except:
            X_prime=np.reshape(X_prime,[M,1])
           
        for i in range(N):
            for j in range(M): 
                x=X[i, :]
                x_prime = X_prime[j, :]
                K[i, j] = self.var_f*np.exp(-0.5*(np.sum( x- x_prime)/self.l)**2)
        return K