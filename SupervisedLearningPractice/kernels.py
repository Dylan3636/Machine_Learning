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
        X = np.array(X)
        X_prime = np.array(X_prime)
        N = np.size(X,0)
        M = np.size(X_prime, 0)        

        if X.ndim != 1:
            X=np.matrix(X)
        else:
            X=np.reshape(X,[N,1])

        if X_prime.ndim > 1:
            X_prime = np.matrix(X_prime)
        else:
            X_prime=np.reshape(X_prime,[M,1])

        K = np.zeros([N, M])
           
        for i in range(N):
            for j in range(M): 
                x=X[i, :]
                x_prime = X_prime[j, :]
                K[i, j] = self.var_f*np.exp(-0.5*(np.sum( x- x_prime)/self.l)**2)
        return K