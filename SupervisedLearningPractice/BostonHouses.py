# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 08:53:25 2017

@author: dylan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearch_CV
# fix random seed for reproducibility
seed = 7

#Part 1 Preprocessing

#loading dataset
dataset = pd.read_csv('housing.csv', delim_whitespace=True, header=None)
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

#Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Part 2 Out of the box Regression models
mses=[]
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression() 
linear_reg.fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)
mse = np.mean((y_pred-y_test)**2)
mses.append(mse)
#Visualization
plt.figure()
plt.title('Linear Regression Results (mse={:04.2f})'.format(mse))
plt.plot(range(0, len(y_test)), np.abs(y_test-y_pred))

plt.legend('Squared Error')
plt.show()

from sklearn.svm import SVR
svm_linear = SVR(kernel = 'linear') 
svm_linear.fit(X_train, y_train)
y_pred = svm_linear.predict(X_test)
mse = np.mean((y_pred-y_test)**2)
mses.append(mse)
#Visualization
plt.figure()
plt.title('SVM Linear Regression Results (mse={:04.2f})'.format(mse))
plt.plot(range(0, len(y_test)), np.abs(y_test-y_pred))

plt.legend('Squared Error')
plt.show()

svm_poly = SVR(kernel = 'poly') 
svm_poly.fit(X_train, y_train)
y_pred = svm_poly.predict(X_test)
mse = np.mean((y_pred-y_test)**2)
mses.append(mse)
#Visualization
plt.figure()
plt.title('SVM Poly Regression Results (mse={:04.2f})'.format(mse))
plt.plot(range(0, len(y_test)), np.abs(y_test-y_pred))

plt.legend('Squared Error')
plt.show()

svm_rbf = SVR(kernel = 'rbf') 
svm_rbf.fit(X_train, y_train)
y_pred = svm_rbf.predict(X_test)
mse = np.mean((y_pred-y_test)**2)
mses.append(mse)
#Visualization
plt.figure()
plt.title('SVM RBF Regression Results (mse={:04.2f})'.format(mse))
plt.plot(range(0, len(y_test)), np.abs(y_test-y_pred))

plt.legend('Squared Error')
plt.show()

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
y_pred = forest.predict(X_test)
mse = np.mean((y_pred-y_test))
mses.append(mse)
#Visualization
plt.figure()
plt.title('SVM Random Forest Regression Results (mse={:04.2f})'.format(mse))
plt.plot(range(0, len(y_test)), np.abs(y_test-y_pred))
plt.legend('Squared Error')
plt.show()

#Neural Networks
from keras.models import Sequential
from keras.layers import Dense
#define base model

def baseline_model():
    # Creating model
    model = Sequential()
    model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compiling model
    model.compile('adam','mean_squared_error')
    return model

BNN = baseline_model() # Baseline Nerual Network
BNN.fit(X_train, y_train, batch_size=10, epochs=100, verbose=0)
y_pred = BNN.predict(X_test)
mse = np.mean((y_pred-y_test.reshape([len(y_test),1]))**2)
mses.append(mse)
# Visualization
plt.figure()
plt.title('Baseline Neural Network Regression Results (mse={:04.2f})'.format(mse))
plt.plot(range(0, len(y_test)), np.abs(y_pred-y_test.reshape([len(y_test),1])))
plt.legend('Squared Error')
plt.show()

# Evaluating different NN models using K-fold cross-validation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
# Evalating baseline model
np.random.seed(seed)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

# evaluate basline model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(13, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# Evaluating deeper model
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))

def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=13, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

# Evaluating wider model
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, y, cv=kfold)
print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))
