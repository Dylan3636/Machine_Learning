# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:08:30 2017

@author: dylan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Part 1 Preprocessing

# loading dataset
from sklearn.datasets import load_iris
dataset = load_iris()
X = dataset.data
y = dataset.target


# Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature Selection

# Recursive Feature Elimination
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    print(dataset.feature_names)
    model = LogisticRegression()
    rfe = RFECV(estimator=model, step =1, cv=5, n_jobs=-1)
    rfe.fit(X_train, y_train)
    print(rfe.support_)
    print(rfe.ranking_)
    print(rfe.grid_scores_)

# Feature importance
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
if __name__ == '__main__':
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y = classifier.feature_importances_
    x = range(len(y))
    fig = plt.figure()
    ax=fig.add_subplot(111)
    ax.bar(x, y, 0.35)
    ax.set_xticks(np.array(range(len(y)))+0.35)
    ax.set_xticklabels(dataset.feature_names)
    ax.set_ylabel('Feature Importance')
    plt.show()
