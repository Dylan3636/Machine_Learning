import numpy as np
import matplotlib.pyplot as plt

#gridsize = 0.1
#x_grid = np.arange(-10, 10, gridsize)
#f_vals = np.cos(x_grid)
#plt.clf()
##plt.plot(x_grid, f_vals, 'b-')
#
#
#
#rbf_1d = lambda x, c, h : np.exp(-((x-c)**2)/h**2)
#plt.plot(x_grid, f_vals, 'r.')
#plt.plot(x_grid, rbf_1d(x_grid, 5, 1), '-b')
#plt.plot(x_grid, rbf_1d(x_grid, -2, 2), '-r')
#plt.show()
#
#y = np.array([1.1, 2.3, 2.9])
#X = np.array([0.8, 1.9, 3.1])

#def rbf(x, c, h):
#   return np.reshape(np.exp(-(x-c)**2/h**2), [len(x),1])

def sig(x, w, b):
   return 1/(1+np.exp(-(np.dot(w,x)+b)))

def phi_poly(x, power):
   return np.concatenate([np.reshape(np.power(x, pw),[len(x),1]) for pw in range(power+1)], axis=1)

def rbf(x, c, h):
    return np.exp(-((x-c)/h)**2)


def rbf_formatter(x, cs, hs):
    if np.shape(hs)[0] == 1:
        hs = np.repeat(hs, len(cs), axis=0)
    return np.concatenate([(rbf(x, c, h)) for (c, h) in zip(cs, hs)], axis=1)

def rbf_mesh(X,bandwidth, h, x_min=None, x_max=None, centres=None):
    centres = get_rbf_centres(X, bandwidth, x_min, x_max) if centres is None else centres
    return [np.concatenate([rbf_formatter(X, cs, [h]) for cs in centres], axis=1), centres]

def get_rbf_centres(x, bandwidth, x_min=None, x_max=None):
    if x_min is None:
        x_maxs = np.max(x, 0)
        x_mins = np.min(x, 0)
    else:
        if np.shape(x_min) == ():
            x_mins=np.repeat(x_min, np.shape(x)[1])
            x_maxs=np.repeat(x_max, np.shape(x)[1])
        else:
            x_maxs = x_max
            x_mins = x_min
    c = get_rbf_centres_between(x_mins, x_maxs, bandwidth)
    return c

def get_rbf_centres_between(low, high, bandwidth):
    if np.shape(low) == ():
        low = [low]
        high = [high]
    return [np.arange(x_min, x_max, bandwidth) for (x_max, x_min) in zip(high, low)]


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
    fit_and_plot((lambda x: rbf(x, [1, 2, 3], [2, 2, 2])), X, y, 'rbf fit')
    plt.legend(loc=4)
    plt.show()





