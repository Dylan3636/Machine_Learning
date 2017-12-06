# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 02:02:06 2017

@author: dylan
"""
import numpy as np
from gaussian_processes import *
class MGP:
    def train(self, K, X, y, kernel=None ,hyperparams=[10,4,2], optimize=False, batch_size=None):
        X=np.matrix(X)
        N = np.size(X,0)
        batch_size = int(N/K) if K is None else K
        ind = np.random.permutation(N)
        gps = []
        for i in range(K):
            XX = X[ind[i*batch_size:(i+1)*batch_size], :]
            yy = y[ind[i*batch_size:(i+1)*batch_size]]
            gp = gaussian_process()
            gp.train(XX, yy, kernel, hyperparams, optimize)
            gps.append(gp)
        self.gps = gps
        
    def batch_predict(self, X):
        val =  np.zeros(np.size(X,0))
        normalizer = np.zeros(np.size(X,0))
        print(len(self.gps))
        for gp in self.gps:
            mus, sigs, ps =gp.batch_predict(X)
            ps = np.exp(ps)
            val += mus*ps
            normalizer += ps
        return val/normalizer
        
        