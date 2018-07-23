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
    D = np.shape(mu)
    x=np.array(x)
    mu = np.reshape(mu, [-1, 1])
    diff = x-mu
    if D==():
        D=1
        cov_inv=1.0/cov
        determinant=cov
        quad = -0.5*diff*cov_inv*diff
    else:
        D= D[0]
        L = np.linalg.cholesky(cov)
        determinant = np.exp(np.sum(np.log(np.diag(L))))
        L_inv = np.linalg.solve(L, np.eye(D))
        cov_inv = np.dot(L_inv.T, L_inv)
        tmp = np.dot(cov_inv,diff)
        quad = -0.5*np.einsum('kj,jk->k...',diff.T,tmp, )
    constant = -0.5*D*np.log(2*np.pi) -0.5*np.log(determinant)
    #print(quad, constant)
    log_pdf = quad+constant
    return log_pdf

def gaussian_random_samples(n, mu, cov):
    if n == 0:
        return
    if np.shape(mu) ==():
        mu = [mu]
    M = np.shape(mu)[0]
    x = np.random.randn(M, n)
    return gaussian_transform(x, mu, cov)

def gaussian_transform(x, mu, cov):
    try:
        A = np.linalg.cholesky(cov)
    except:
        A = np.sqrt(cov)
    rot_x = np.dot(A,x)
    new_x = np.add(rot_x, np.reshape(mu, [-1,1]))
    return new_x

def plot_2D_gaussian_scatter_PCA(n, mu, cov):
    if type(n) is int:
        M = np.shape(mu)[0]
        smps = np.random.randn(M, n)  
        U,S,V = np.linalg.svd(cov)
        A =np.dot(U,np.diag(np.sqrt(S)))
        rot_x = np.dot(np.dot(A,smps).T,V[:,0:2])
        x = np.add(rot_x, np.reshape(np.dot(np.transpose(mu), V[:,0:2]), [1,-1]))
        plt.scatter(x[:,0],x[:,1])
    else:
        smps = n
        U,S,V = np.linalg.svd(smps)
        A =np.dot(U,np.diag(np.sqrt(S)))
        cntrd = np.subtract(smps,np.reshape(mu, [-1,1]))
        rot_x = np.dot(np.dot(A,cntrd).T,V[:,0:2])
        x= np.add(np.dot(mu.T, V[:,0:2]).reshape([1,-1]), A)
        plt.scatter(x[:,0],x[:,1])
        
def plot_1D_gaussian_line_PCA(n, mu, cov):
    
    if type(n) is int:
        M = np.shape(mu)[0]
        smps = np.random.randn(M, n)  
        U,S,V = np.linalg.svd(cov)
        A =np.dot(U,np.diag(np.sqrt(S)))
        rot_x = np.dot(np.dot(A,smps).T,V[:,0:1])
        mu = np.reshape(np.dot(np.transpose(mu), V[:,0:1]), [1,-1])
        x = np.add(rot_x, mu).ravel()
        x = np.sort(x)
        s = np.std(x)
        mu = mu.ravel()[0]
        p = np.exp(log_gaussian_pdf(x, mu, s)[0])
        print(mu, s)
        plt.plot(x.ravel(),p)
    else:
        smps = n
        U,S,V = np.linalg.svd(smps)
        A = V[:,0:1]
        x= np.add(np.dot(np.subtract(smps,np.reshape(mu, [1,-1])).T, A), np.dot(mu.T, A).reshape([1,-1])).ravel()
        x = np.sort(x)
        s = np.std(x)
        mu = mu.ravel()[0]
        p = np.exp(log_gaussian_pdf(x, mu, s)[0])
        plt.plot(x.ravel(),p)

    

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

def get_sigma_points(mu, cov, alpha, kappa):
    n = np.size(mu, 0)
    chai = np.zeros([2*n + 1, n])
    chai[0, :] = mu
    L=np.linalg.cholesky(cov)
    for i in range(2*n):
        lmbda = (alpha**2)*(n+kappa)-n
        if i<n:
            chai[i+1, :] = mu + np.sqrt(n + lmbda)*L[i,:]
        else:
            chai[i+1, :] = mu - np.sqrt(n + lmbda)*L[i-n,:]
    return chai

def plot_2D_sigma_points(mu, cov, alpha, kappa):
    chai = get_sigma_points(mu, cov, alpha, kappa)
    plt.scatter(chai[:, 0], chai[:, 1])
    
def get_2D_confidence_region(n, mu, cov, alpha=0.05):
    r = np.sqrt(-2*np.log(alpha))
    #sigmas = np.array(cov).flatten()
    #rho = sigmas[2]/np.sqrt(sigmas[1]*sigmas[-1])
    M = np.linalg.cholesky(cov)
    thetas = np.linspace(0, 2*np.pi, n)
    pts = np.dot(r*np.column_stack([np.cos(thetas), np.sin(thetas)]), M) + mu
    return pts

def plot_2D_confidence_interval(n, mu, cov, alpha=0.05):
    pts = get_2D_confidence_region(n, mu, cov, alpha)
    plt.plot(pts[:, 0], pts[:, 1])
