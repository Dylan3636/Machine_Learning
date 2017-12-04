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
    
    def __init__(self, X, y, kernel):
        self.X=X
        self.y=y
        self.K = kernel(X, X)
    def calculate_gp_parameters(self, X_prime, X=None, y=None, sig_y=0.1, kernel=None):
        var_y = sig_y**2
        K = kernel(X, X)
        K_prime_L = kernel(X_prime, X)
        K_prime_R = kernel(X, X_prime)
        K_prime_prime = kernel(X_prime, X_prime)
        
        N = np.shape(y)[0]
        K_y = K + var_y*np.eye(N)
        tmp_y = np.linalg.solve(K_y, y);
        mu = np.dot(K_prime_L, tmp_y);
        del(tmp_y)
        tmp_R = np.linalg.solve(K_y, K_prime_R);
        cov = K_prime_prime - np.dot(K_prime_L, tmp_R);
        return mu, cov
    
    def get_gp_error_bars(self,alpha,mu, var):
        upper = mu + var
        lower = mu - var
        return upper, lower
    
    def plot_gp_samples(x, n, mean, cov, color=[0.5,0.5,0.5]):
        f = gaussian_random_samples(n, mean, cov);
        for i in range(n):
           plt.plot(x, f[:,i], color=color)
  
    def cost_func(self, x):
        theta_f, theta_l, theta_y = x
        print(theta_f, theta_l, theta_y)
        N=np.size(self.K,0)
        self.kernel = rbf_kernel(np.sqrt(np.exp(theta_f)), np.sqrt(np.exp(theta_l))).kernel
        self.K = self.kernel(self.X, self.X)
        K_y = self.K + np.exp(theta_y)*np.eye(N)
        
        alpha = np.linalg.solve(K_y, self.y)

        fit_term = -0.5*np.dot(self.y.T, alpha)
        complexity_term = -0.5*np.log(np.linalg.det(K_y))
        constant = -0.5*N*np.log(2*np.pi)
        
        cost = fit_term + complexity_term + constant
        
        K_y_inv = np.linalg.inv(K_y)
        deriv_main = lambda x: -0.5*np.trace(np.dot((np.dot(alpha, alpha.T)-K_y_inv),x))
        
        deriv_f = 0*self.K
        deriv_l = 0*self.K*(np.log(self.K)-theta_f)
        deriv_y = np.exp(theta_y)*np.eye(N)
        gradients = np.array([deriv_main(deriv_f), deriv_main(deriv_l), deriv_main(deriv_y)] )       
        print(fit_term , complexity_term , constant)
        print(-cost, gradients)
        return -cost#, (gradients)

    def train(self, init=[0.9,4,2]):
        opt = {'maxiter': 2000, 'disp': True,'eps' : 1e-1}
        init = np.log(init)
        params = minimize(self.cost_func, init, method='L-BFGS-B', jac=False, options=opt)
        print(params)
        return np.sqrt(np.exp(params.x))


    
    def cost_deriv(self, x):
        theta_f, theta_l, theta_y = x
        n=np.size(self.K,0)
        
        
def demo(sig_f = 1.39,l=1.78, sig_y = 0.55,X=None, y =None):
    n = 10
    n_w= 0
    
    if X is None:
        X = np.random.rand(n)
        y =  2*X**5 + 1*np.random.rand(n)
    
    
    xs = np.linspace(-5, 5, 1000)
    #ys = 2*xs**5 + 0*np.rand([np.size(xs,0), 1]);
    
    X_prime = xs
    kernel = rbf_kernel(sig_f, l).kernel
    gp = gaussian_process( X, y, kernel)
    fs = np.zeros([1000, n_w])
    mu = np.zeros(1000)
    cv = np.zeros(1000)
    for i in range(1000):
        m, s = gp.calculate_gp_parameters([X_prime[i]], X, y,sig_y, kernel);
        fs[i,:]=(gaussian_random_samples(n_w, m, s))
        mu[i] = m
        cv[i] = s
    for j in range(n_w):
        plt.plot(xs, fs[:,j], color=[0.5,0.5,0.5])
    [upper_ys, lower_ys]=gp.get_gp_error_bars(1, mu, cv);
    p1=plt.plot(xs,mu,'black');
        #shadedErrorBar(xs,mu,[upper_ys-mu])
    p2=plt.plot(xs, upper_ys, 'red');
    p3=plt.plot(xs, lower_ys, 'blue');
    #plot_gp_samples(xs,n_w, mu, cv, [0.5,0.5,0.5])
        
    plt.scatter(X, y, c='r');
    #plt.legend([p1,p2,p3],{'mean', 'upper bound', 'lower bound'})
    plt.show()

def optimize_demo():
        n = 10
        X = np.concatenate([np.linspace(-4, -2, n/2),np.linspace(2, 4, n/2)])
        y =  2*X**3 + 1*np.random.randn(n)
        kernel = rbf_kernel(1, 1).kernel
        gp = gaussian_process(X,y, kernel)
        params = gp.train([300,4,2])
        print(params)
        demo(params[0],params[1],params[2],X,y)

if __name__ == '__main__':
    optimize_demo()