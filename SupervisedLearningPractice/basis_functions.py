import numpy as np
import matplotlib.pyplot as plt

gridsize = 0.1
x_grid = np.arange(-10, 10, gridsize)
f_vals = np.cos(x_grid)
plt.clf()
#plt.plot(x_grid, f_vals, 'b-')



rbf_1d = lambda x, c, h : np.exp(-((x-c)**2)/h**2)
plt.plot(x_grid, f_vals, 'r.')
plt.plot(x_grid, rbf_1d(x_grid, 5, 1), '-b')
plt.plot(x_grid, rbf_1d(x_grid, -2, 2), '-r')
plt.show()

y = np.array([1.1, 2.3, 2.9])
X = np.array([0.8, 1.9, 3.1])

def rbf(x, c, h):
   return np.reshape(np.exp(-(x-c)**2/h**2), [len(x),1])

def sig(x, v, b):
   return 1/(1+np.exp(-(np.dot(v,x)+b)))

def phi_poly(x, power):
   return np.concatenate([np.reshape(np.power(x, pw),[len(x),1]) for pw in range(power+1)], axis=1)

def phi_rbf(x, cs, hs):
   return np.concatenate([rbf(x, ch[0], ch[1]) for ch in zip(cs, hs)], axis=1)

def fit_and_plot(phi_func, X, y,label=None ):
   Phi = phi_func(X)
   print(Phi)
   w = np.linalg.lstsq(Phi, y)[0]
   x_grid = np.arange(0, 4, 0.01)
   phi_grid = phi_func(x_grid)
   y_grid = np.dot(phi_grid, w)
   plt.plot(x_grid, y_grid, label=label)  

plt.plot(X, y, 'r.', label='data')
fit_and_plot((lambda x: phi_poly(x, 1)), X, y, 'linear fit')
fit_and_plot((lambda x: phi_poly(x, 2)), X, y, 'quadratic fit')
fit_and_plot((lambda x: phi_rbf(x, [1, 2, 3], [2, 2, 2])), X, y, 'rbf fit')
plt.legend(loc=4)
plt.show()





