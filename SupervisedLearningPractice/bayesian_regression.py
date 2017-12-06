# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 20:55:42 2017

@author: dylan
"""
import numpy as np
from gaussians import *
from basis_functions import *
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class bayesian_regression:
        
    def get_weight_posterior_paramters(self, X, y, sig_y=0.1, w_0=None, sig_w=1):
        var_y = sig_y**2
        var_w = sig_w**2
        n, d = np.shape(X)
        d+=1
        phi = np.concatenate([X, np.ones([n, 1])], axis=1)
        if w_0 is None:
            w_0 = np.zeros(d)
        try:
            var_w = np.diag(var_w)
        except:
            pass
        is_zero= True
        V_0 = var_w*np.eye(d)
        Inv = np.linalg.solve(V_0, var_y*np.eye(d))
        V_n = np.linalg.solve( Inv + np.dot(phi.T, phi), var_y*np.eye(d))
        if is_zero:
            w_n = (1/var_y)*np.dot(np.dot(V_n, phi.T), y)
        else:
            Inv_V = np.dot(V_n, np.linalg.solve(V_0, np.eye(d) ))
            w_n = np.dot(np.dot(V_n,Inv_V), w_0.ravel()) + (1/var_y)*np.dot(np.dot(V_n, phi.T), y)           
        return w_n, V_n
    
    def get_prediction_parameters(self, x, sig_y, w_n, V_n):
        var_y = sig_y**2
        mew = np.dot(x, w_n)
        cov = np.dot(np.dot(x.T, V_n),x) + var_y
        return mew, cov
    
    def calculate_nll(self, sig_y, w_n, V_n):
        var_y = sig_y**2
        n = np.shape(self.X)[0]
        log_likelihood=0
        try:
            for i in range(n):
                x = self.X[i, :]
                x=np.append(x, 1)
                mew, cov = self.get_prediction_parameters(x,var_y, w_n, V_n)
                log_pdf = log_gaussian_pdf(self.y[i], mew, cov)[0][0]
                if log_pdf >0:
                    log_likelihood = -1e100
                    break
                #print(self.y[i],mew, cov, log_pdf)
                log_likelihood += log_pdf
        except:
            log_likelihood = -1e100
        return np.nanmin([-log_likelihood, 1e100])
    def train(self, X, y, w_0=None, hyperparams=[2],optimize=False):
        self.X = X
        self.y = y
        try:
            n, d = np.shape(X)
        except:
            d=1
            n = np.size(X, 0)
        if w_0 is None:
            w_0 = np.zeros(d)
        self.w_0=np.zeros(d)
        self.hyperparams = hyperparams
        self.w_N, self.V_N = self.get_weight_posterior_paramters( X, y, sig_y=hyperparams[0], w_0=w_0, sig_w=hyperparams[1])
        if optimize:
           opt = {'maxiter': 3000, 'disp': True}
           init = np.log(hyperparams)
           params = minimize(self.cost_func, init, method='L-BFGS-B', jac=False, options=opt)
           print(params)
           self.hyperparams=np.exp(params.x)
           print(self.hyperparams)
           self.w_N, self.V_N = self.get_weight_posterior_paramters( X, y, sig_y=self.hyperparams[0], w_0=w_0, sig_w=self.hyperparams[1])
        return self.w_N, self.V_N
         
            

    def get_error_bars(self, alpha,xs, w_N, V_N, sigma_y):
        N = np.size(xs,0)
        upper_ys = np.zeros([N, 1])
        lower_ys = np.zeros([N, 1])
        
        for i in range(N):
            x=xs[i,:]
            phi = np.concatenate([x, [1]], 0)
            mu, sigma = self.get_prediction_parameters(phi, sigma_y, w_N, V_N)
            upper_ys[i] = mu + alpha*sigma
            lower_ys[i] = mu - alpha*sigma
        return upper_ys, lower_ys
    def cost_func(self, params):
        print(params)
        params = np.exp(0.5*params)
        try:
            self.w_N, self.V_N = self.get_weight_posterior_paramters( self.X, self.y, sig_y=params[0], w_0=self.w_0, sig_w=params[1])
            cost = self.calculate_nll(params[0], self.w_N, self.V_N)
        except:
            cost=1e100
        print(cost)
        return cost
    

def plot_line_samples(phi_xs, x, mu, cov, n, color):
    ws = gaussian_random_samples(n, mu, cov)
    X = np.concatenate([phi_xs, np.ones([np.size(x, 0), 1])], 1)
    for i in range(n):
       y = np.dot(X,ws[:, i])
       plt.plot(x, y,color=color)
    

def br_plots(sig_y=2, sig_w = 100, X=None, y=None):
    n = 10
    n_w= 5
    bandwidth=0.05
    h=5;
        
    x = np.array([-12, -10,-8,-6,-4, 4,6,8,10,12])#np.concatenate([np.linspace(-3, -2, n/2),np.linspace(2, 3, n/2)])
    X = x if X is None else X
    y =  np.sin(X) + 1*np.random.randn(n) if y is None else y
    br = bayesian_regression()
    #params = gp.train(X,y, hyperparams = [10,4,2], optimize=True)
    #print(params)
    #sig_f, l, sig_y = params
    phi_X, cs = rbf_mesh(np.reshape(X, [n, 1]), bandwidth, h)
    w_0 = np.concatenate([np.zeros([np.size(phi_X, 1), 1]), [[0]]], 0)
    w_N, V_N = br.train(phi_X, y, None, [0.2,10], True)

    xs = np.linspace(-12, 12, 1000)
    phi_xs,_ = rbf_mesh(np.reshape(xs, [1000, 1]), bandwidth, h, centres = cs)
    

    upper_ys, lower_ys = br.get_error_bars(1, phi_xs, w_N, V_N, br.hyperparams[0])
    p2=plt.plot(xs, upper_ys, 'red')
    p1=plt.plot(xs,np.dot(np.concatenate([phi_xs, np.ones([1000, 1])], axis=1), w_N),'black');
        #shadedErrorBar(xs,mu,[upper_ys-mu])
    p3=plt.plot(xs, lower_ys, 'blue')
    plot_line_samples(phi_xs, xs, w_N, V_N, n_w, None)
        
    plt.scatter(X, y, c='r')
    plt.legend({'mean', 'upper bound', 'lower bound'})
    plt.figure()
    plot_1D_gaussian_line_PCA(1000, w_N, V_N)
    tmp=np.sort(gaussian_random_samples(1000, np.mean(w_N), np.std(w_N)))
    print(np.mean(w_N), np.std(w_N))
    plt.plot(tmp.ravel(), np.exp(log_gaussian_pdf(tmp,np.mean(w_N), np.std(w_N) )[0]))

    plt.show()

if __name__ == '__main__':
    br_plots()