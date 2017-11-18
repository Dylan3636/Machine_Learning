# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:10:30 2017

@author: dylan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

seed = 7
np.random.seed(seed)
# Part 1 Preprocessing

# loading dataset
dataset = pd.read_csv('No-show.csv')

# Cleaning dataset
dataset = dataset.rename(columns = {'Alcoolism': 'Alcoholism', 'ApointmentData':'AppointmentData'})
dataset['AwaitingTime'] = dataset['AwaitingTime'].apply(abs)
dataset = dataset[dataset['Age'].apply(lambda x: x>0 and x<96).values]
dataset = dataset[(dataset['AwaitingTime']<=150).values]

y = dataset['Status'].values
dataset=dataset.drop(['AppointmentData', 'AppointmentRegistration','Status' ], 1)
X = dataset.values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
gender_enc = LabelEncoder()
X[:, 1] = gender_enc.fit_transform(X[:, 1])
dow_enc = LabelEncoder()
X[:, 2] = dow_enc.fit_transform(X[:, 2])
# onehotencoder = OneHotEncoder(categorical_features = [2])
# X = onehotencoder.fit_transform(X).toarray()
# X = X[:, 1:]
show_enc = LabelEncoder()
y = show_enc.fit_transform(y)

# Sample part of dataset
from sklearn.model_selection import train_test_split
# X,_,y,_ = train_test_split(X, y, train_size = 0.5, random_state=0)


#Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Building models
accuracies={}

#Logistic Regression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

from sklearn.naive_bayes import MultinomialNB

#classifier = LogisticRegressionCV(5, cv=5, solver='sag', n_jobs=-1)
classifier = MultinomialNB()
if __name__ == '__main__':
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    accuracy = sum(y_pred==y_test)/len(y_test)
    accuracies['Logistic Regression'] = accuracy
    print(accuracy)



if __name__ == '__main__':
    model = MultinomialNB()
    rfe = RFECV(estimator=model, step =1, cv=5, n_jobs=-1)
    rfe.fit(X_train, y_train)
    print(rfe.support_)
    print(rfe.ranking_)
    print(rfe.grid_scores_)


# from sklearn.naive_bayes import MultinomialNB
# classifier = MultinomialNB()
#
#
# if __name__ == '__main__':
#     classifier.fit(X_train, y_train)
#
#     # Predicting the Test set results
#     y_pred = classifier.predict(X_test)
#     y_pred = (y_pred > 0.5)
#     accuracy = sum(y_pred==y_test)/len(y_test)
#     accuracies['Logistic Regression'] = accuracy
#     print(accuracy)

# SVM
# from sklearn.svm import SVC
# parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear'] },
#                 {'C': [1, 10, 100, 1000], 'kernel':['rbf'], 'gamma': [0.1,0.01,0.001] },
#                 {'C': [1, 10, 100, 1000], 'kernel': ['poly'],'degree': range(1,4) }
# ]
# if __name__ == '__main__':
#     grid_cv = GridSearchCV(SVC(), parameters, n_jobs =-1, scoring='accuracy', cv=10)
#     grid_cv.fit(X_train, y_train)
#
#     # Predicting the Test set results
#     y_pred = classifier.predict(X_test)
#     y_pred = (y_pred > 0.5)
#     accuracy = sum(y_pred==y_test)/len(y_test)
#     print(accuracy)
#     accuracies['Logistic Regression'] = accuracy
#
#     # Making the Confusion Matrix
#     cm = confusion_matrix(y_test, y_pred)
#     print(cm)

# Random Forest
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier()
# parameters = [{'n_estimators': [75,100,125]}]
# if __name__ == '__main__':
#     grid_cv = GridSearchCV(classifier, parameters, n_jobs =-1, scoring='accuracy', cv=5, verbose=2)
#     grid_cv.fit(X_train, y_train)
#     print (grid_cv.best_score_)
#     print (grid_cv.best_params_)
#     classifier=grid_cv.best_estimator_
#     Y = classifier.feature_importances_
#     x = range(len(Y))
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.bar(x, Y, 0.8)
#     ax.set_xticks(np.array(range(len(Y))) + 0.8)
#     ax.set_xticklabels(dataset.columns.values)
#     ax.set_ylabel('Feature Importance')
#     plt.show()
#
# from sklearn.ensemble import RandomForestClassifier
# if __name__ == '__main__':
#     classifier = RandomForestClassifier()
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     print(sum(y_pred==y_test)/len(y_test))
#     Y = classifier.feature_importances_
#     x = range(len(Y))
#     fig = plt.figure()
#     ax=fig.add_subplot(111)
#     ax.bar(x, Y, 0.8)
#     ax.set_xticks(np.array(range(len(Y)))+0.8)
#     ax.set_xticklabels(dataset.columns.values)
#     ax.set_ylabel('Feature Importance')
#     plt.show()
#     X_train= X_train[:,Y>0.4]
#     X_test = X_test[:, Y>0.4]
#     print(dataset.columns.values[Y>0.4])
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
#     print(sum(y_pred == y_test) / len(y_test))
#
# classifier = LogisticRegressionCV(5, cv=5, solver='sag', n_jobs=-1)
#
# if __name__ == '__main__':
#     classifier.fit(X_train, y_train)
#
#     # Predicting the Test set results
#     y_pred = classifier.predict(X_test)
#     y_pred = (y_pred > 0.5)
#     accuracy = sum(y_pred==y_test)/len(y_test)
#     accuracies['Logistic Regression'] = accuracy
#     print(accuracy)
#
# classifier = MultinomialNB()
# if __name__ == '__main__':
#     classifier.fit(X_train, y_train)
#
#     # Predicting the Test set results
#     y_pred = classifier.predict(X_test)
#     y_pred = (y_pred > 0.5)
#     accuracy = sum(y_pred==y_test)/len(y_test)
#     accuracies['Logistic Regression'] = accuracy
#     print(accuracy)

# classifier = RandomForestClassifier()
# parameters = [{'n_estimators': [1,2,3,4]}]
# if __name__ == '__main__':
#     grid_cv = GridSearchCV(classifier, parameters, n_jobs =-1, scoring='accuracy', cv=5, verbose=2)
#     grid_cv.fit(X_train, y_train)
#     print (grid_cv.best_score_)
#     print (grid_cv.best_params_)


#
# #Neural Network
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.wrappers.scikit_learn import KerasClassifier
#
# def base_model():
#     model = Sequential()
#     model.add(Dense(2, input_dim=2, activation='relu'))
#     model.add(Dense(1, activation = 'sigmoid'))
#     model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#     return model
#
# model = KerasClassifier(build_fn=base_model, verbose=0)
#
#
# batch_size = [5, 10, 50, 100]
# epochs = [5, 10, 25]
# parameters = dict(batch_size = batch_size, epochs= epochs)
# if __name__ == '__main__':
#     grid = GridSearchCV(estimator=model, param_grid=parameters, n_jobs= 6, verbose = 0)
#     grid_result = grid.fit(X_train, y_train)
#     # summarize results
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))
# Best batch size: 50 and epochs: 10
#
# def create_model(optimizer='adam'):
#     model = Sequential()
#     model.add(Dense(2, input_dim=2, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model
#
# model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=50, verbose=1)
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# param_grid = dict(optimizer=optimizer)
# if __name__ == '__main__':
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv= 5)
#     grid_result = grid.fit(X_train, y_train)
#     # summarize results
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))

#Best Optimizer: RMSprop
#
# def create_model(init_mode='uniform'):
#     model = Sequential()
#     model.add(Dense(17, input_dim=17, activation='relu', kernel_initializer=init_mode))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
#     return model
#
# model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=50, verbose=0)
# init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
# param_grid = dict(init_mode=init_mode)
#
# if __name__ == '__main__':
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv =5)
#     grid_result = grid.fit(X, y)
#     # summarize results
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))
# Best he_uniform
#
# def create_model(activation ='relu' ):
#     model = Sequential()
#     model.add(Dense(17, input_dim=17, activation=activation, kernel_initializer='he_uniform'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
#     return model
#
# model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=50, verbose=0)
# activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
# param_grid = dict(activation=activation)
# if __name__ == '__main__':
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#     grid_result = grid.fit(X, y)
#     # summarize results
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))


# def create_model(dropout_rate=0.0):
#     # create model
#     model = Sequential()
#     model.add(Dense(17, input_dim=17, kernel_initializer='uniform', activation='softmax'))
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=100, verbose=0)
# # define the grid search parameters
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# param_grid = dict(dropout_rate=dropout_rate)
# if __name__ == '__main__':
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#     grid_result = grid.fit(X, y)
#     # summarize results
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))

#
# def create_model(neurons=17, num_hidden_layers = 1):
# 	# create model
# 	model = Sequential()
# 	for _ in range(num_hidden_layers):
# 	    model.add(Dense(neurons, input_dim=17, kernel_initializer='uniform', activation='relu'))
# 	#model.add(Dropout(0.1))
# 	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
# 	return model
#
# model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=100, verbose=0)
# # define the grid search parameters
# neurons = [9, 17, 24, 30]
# num_hidden_layers = [1, 2, 3, 4, 5]
# param_grid = dict(neurons=neurons, num_hidden_layers = num_hidden_layers)
# if __name__ == '__main__':
#     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#     grid_result = grid.fit(X, y)
#     # summarize results
#     print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#     means = grid_result.cv_results_['mean_test_score']
#     stds = grid_result.cv_results_['std_test_score']
#     params = grid_result.cv_results_['params']
#     for mean, stdev, param in zip(means, stds, params):
#         print("%f (%f) with: %r" % (mean, stdev, param))

#
# if __name__ == '__main__':
#     model = Sequential()
#     model.add(Dense(2, input_dim=2, kernel_initializer='uniform', activation='relu'))
#     model.add(Dense(1, kernel_initializer='uniform', activation= 'sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=5, batch_size=100)
#     y_pred = model.predict(X_test)
#     print(y_pred)
#     y_pred = y_pred[0]
#     print(y_pred)
#     y_pred = (y_pred > 0.5)
#     accuracy = sum(y_pred==y_test)/len(y_test)
#     print(accuracy)
#
