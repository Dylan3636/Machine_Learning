# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 00:08:40 2017

@author: dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from gaussians import *
from kernels import *
from scipy.optimize import minimize

class gaussian_process:
    
    def calculate_gp_parameters(self, X_prime, sig_y=None):
        X = self.X
        y = self.y
        kernel = self.kernel
        self.hyperparams[2] = self.hyperparams[2] if sig_y is None else sig_y
        var_y = self.hyperparams[2]**2
        K = self.K
        K_prime = kernel(X, X_prime)
        K_prime_prime = kernel(X_prime, X_prime)
        
        N = np.shape(y)[0]
        K_y = K + var_y*np.eye(N)
        L = np.linalg.cholesky(K_y)        
        alpha = np.linalg.solve(L.T,np.linalg.solve(L, self.y))
        mu = np.dot(K_prime.T, alpha)
        del(alpha)
        v = np.linalg.solve(L, K_prime)
        cov = K_prime_prime - np.dot(v.T, v)
        return mu, cov
    
    
    def cost_func(self, x):
        theta_f, theta_l, theta_y = x
        #print(theta_f, theta_l, theta_y)
        N=np.size(self.K,0)
        self.kernel = rbf_kernel(np.sqrt(np.exp(theta_f)), np.sqrt(np.exp(theta_l))).kernel
        self.K = self.kernel(self.X, self.X)
        K_y = self.K + np.exp(theta_y)*np.eye(N)
        L = np.linalg.cholesky(K_y)
        alpha = np.linalg.solve(L.T,np.linalg.solve(L, self.y))

        fit_term = -0.5*np.dot(self.y.T, alpha)
        complexity_term = -0.5*np.log(np.sum(np.diag(L)))
        constant = -0.5*N*np.log(2*np.pi)
        
        cost = fit_term + complexity_term + constant
#        
#        K_y_inv = np.linalg.inv(K_y)
#        deriv_main = lambda x: -0.5*np.trace(np.dot((np.dot(alpha, alpha.T)-K_y_inv),x))
#        
#        deriv_f = 0*self.K
#        deriv_l = 0*self.K*(np.log(self.K)-theta_f)
#        deriv_y = np.exp(theta_y)*np.eye(N)
#        gradients = np.array([deriv_main(deriv_f), deriv_main(deriv_l), deriv_main(deriv_y)] )       
        #print(fit_term , complexity_term , constant)
#        print(-cost, gradients)
        return -cost#, (gradients)

    def train(self, X, y, kernel=None, hyperparams=[5,2,1], optimize=False):
        self.X = X
        self.y = y
        self.kernel = rbf_kernel(hyperparams[0], hyperparams[1]).kernel if kernel is None else kernel
        self.K = self.kernel(X, X)
        if optimize:
            opt = {'maxiter': 2000, 'disp': True}
            init = np.log(hyperparams)
            params = minimize(self.cost_func, init, method='L-BFGS-B', jac=False, options=opt)
            hyperparams = np.sqrt(np.exp(params.x))
            print(params)
            self.hyperparams = hyperparams
            self.kernel = rbf_kernel(hyperparams[0], hyperparams[1]).kernel
        self.K = self.kernel(X, X)
        return hyperparams

    def batch_predict(self, X):
        N,D=np.shape(np.matrix(X))
        mus = np.zeros([N])
        sigs = np.zeros([N])
        log_pdfs = np.zeros([N])
        for i in range(N):
            m,s = self.calculate_gp_parameters(X[i,:])
            mus[i] = m
            sigs[i] = s
            log_pdfs[i] = log_gaussian_pdf(m,m,s)
        return mus, sigs, log_pdfs
    
    def cost_deriv(self, x):
        theta_f, theta_l, theta_y = x
        n=np.size(self.K,0)
        
def get_gp_error_bars(alpha,mu, var):
        error = alpha*np.sqrt(var)
        upper = mu + error
        lower = mu - error
        return upper, lower
    
def plot_gp_samples(x, n, mean, cov, color=[0.5,0.5,0.5]):
        f = gaussian_random_samples(n, mean, cov);
        for i in range(n):
           plt.plot(x, f[:,i], color=color)
        
def gp_plots(sig_f = 1.39,l=1.78, sig_y = 0.55, X=None, y=None):
    n = 10
    n_w= 0
    m=2000
    x = np.array([-12, -10,-8,-6,-4, 4,6,8,10,12])#np.concatenate([np.linspace(-3, -2, n/2),np.linspace(2, 3, n/2)])
    X = x if X is None else X
    y =  np.sin(X) + 5*np.random.randn(n) if y is None else y
    gp = gaussian_process()
    params = gp.train(X,y, hyperparams = [10,10,2], optimize=True)
    print(params)
    sig_f, l, sig_y = params
    
    xs = np.linspace(-24, 24, m)
    #ys = 2*xs**5 + 0*np.rand([np.size(xs,0), 1]);
    
    X_prime = xs
    fs = np.zeros([m, n_w])
    mu = np.zeros(m)
    cv = np.zeros(m)
#    mu, cv = gp.calculate_gp_parameters(X_prime)

    for i in range(m):
        m, s = gp.calculate_gp_parameters([X_prime[i]]);
        fs[i,:]=(gaussian_random_samples(n_w, m, s))
        mu[i] = m
        cv[i] = s
    
    plt.scatter(X, y, c='r')
    p1=plt.plot(xs,mu,'black');
    [upper_ys, lower_ys]=get_gp_error_bars(1, mu,cv)
    p2=plt.plot(xs, upper_ys, 'red')
    p3=plt.plot(xs, lower_ys, 'blue')
    for j in range(n_w):
        plt.plot(xs, fs[:,j], color=None)
    
    #plot_gp_samples(xs,n_w, mu, cv, [0.5,0.5,0.5])
        
    plt.legend({'mean', 'upper bound', 'lower bound'})
    plt.show()

if __name__ == '__main__':
    gp_plots()