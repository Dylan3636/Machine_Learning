# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:02:49 2017

@author: dylan
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm

def log_gaussian_pdf(x, mu, cov):
    x=np.array(x)
    mu = np.reshape(mu, [-1, 1])
    D = np.shape(x)
    diff = x-mu
    if D==():
        D=1
        cov_inv=1.0/cov
        determinant=cov
        quad = -0.5*diff*cov_inv*diff
    else:
        D= D[0]
        L = np.linalg.cholesky(cov)
        determinant = np.prod(np.diag(L))
        L_inv = np.linalg.solve(L, np.eye(D))
        cov_inv = np.dot(L_inv.T, L_inv)
        tmp = np.dot(cov_inv,diff)
        quad = -0.5*np.einsum('kj,jk->k...',diff.T,tmp, );
    constant = -0.5*D*np.log(2*np.pi) +0.5*np.log(determinant);
    log_pdf = quad+constant
    return log_pdf

def gaussian_random_samples(n, mu, cov):
    if np.shape(mu) ==():
        mu = [mu]
    M = np.shape(mu)[0]
    x = np.random.randn(M, n)
    return gaussian_transform(x, mu, cov)

def gaussian_transform(x, mu, cov):
    U,S,V = np.linalg.svd(cov);
    A = np.dot(U,np.sqrt(S))
    rot_x = np.dot(A,x);
    new_x = np.add(rot_x, np.reshape(mu, [-1,1]));
    return new_x

def plot_2D_gaussian_contour(npts, mu, cov):
    x = np.random.uniform(-2,2,npts)
    y = np.random.uniform(-2,2,npts)
    z = log_gaussian_pdf(np.reshape([x,y],[2,npts]), mu, cov)
    xi = np.linspace(-2.1,2.1,100)
    yi = np.linspace(-2.1,2.1,100)
    
    ## grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi,yi,zi,6,linewidths=0.5,colors='k')
    CS = plt.contourf(xi,yi,zi,6,cmap=cm.Greys_r)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.title('griddata test (%d points)' % npts)
    plt.show()

def plot_2D_gaussian_scatter(n, mu, cov):
    x = gaussian_random_samples(n, mu, cov)    
    plt.scatter(x[0,:],x[1,:])
