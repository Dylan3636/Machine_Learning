# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 01:57:24 2017

@author: dylan
"""

import numpy as np
import pandas as pd


def clean_state(state):
    tmp = state.split()[1::]
    tmp[-1] = state[-3] +'.'
    tmp= np.array(tmp, dtype=float)
    orientation = tmp[-3::]
    one_hot = np.zeros(8)
    index = int(orientation[0] + 2*orientation[1] + 4*orientation[2])
    one_hot[index] = 1
    return np.array(list(tmp[0:-3]) + list(one_hot), dtype=int)
     
def sum_rewards(episodes):
    episodes = episodes.copy()
    episodes['Return'] = None
    
    for name, episode in episodes.groupby('Episode'):
        print(name)
        rows = episode.index.values
        episodes.set_value(rows, 'Total Return', sum(episode['Reward'])*np.ones(len(episode)) )
    return episodes
 
# Preprocessing
data = pd.read_csv('datasets\log_data_episode_30000.csv')
tmp = data['Current State']
data['Current State'] = data['Reward']
dic = {100: 1, -10: -0.2, -1: -0.1 }
data['Reward'] = tmp.apply(lambda x: dic[x])
del(tmp)
data['Previous State'] = data['Previous State'].apply(clean_state).values
data = data.groupby('Episode').apply(sum_rewards)

X = np.array(list(data['Previous State'].values))
R = data['Total Return'].values

# Neural Network imports
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K


def init_tensorflow_model(input_shape):
    tf.reset_default_graph()
    
    X_holder = tf.placeholder('float', input_shape, name='X')
    R_holder = tf.placeholder('float', (input_shape[0], 1), name='R')
    
    W1 = tf.get_variable("W1", (input_shape[1], 300), initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b1", (1,300), initializer=tf.constant_initializer(0.0))
    out1 = tf.matmul(X_holder,W1) + b1
    
    output = tf.get_variable("output", (300,3), initializer=tf.random_normal_initializer())
    b2 = tf.get_variable("b2", (1,3), initializer=tf.constant_initializer(0.0))
    y_pred = tf.matmul(out1, output) +b2
    
    tmp = tf.exp(y_pred)
    y_pred = tf.divide(tmp,tf.reduce_sum(tmp)) 
    loss = tf.multiply(tf.log(y_pred), R_holder)
    update_operation = tf.train.AdamOptimizer().minimize(loss)
    return update_operation, X_holder, R_holder

def train_tens_model(optimizer,X_holder, r_holder, X, R, batch_size, iterations):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(iterations):
            indices = np.random.choice(len(X), size=batch_size,replace = False )
            X_batch, R_batch = X[indices], np.reshape(R[indices],(batch_size, 1))
            sess.run([optimizer], feed_dict={X_holder:X_batch, r_holder:R_batch})

#opt, X_holder, r_holder = init_tensorflow_model((1,21))
#train_model(opt,X_holder,r_holder, np.array(list(X['Previous State'].values)), X['Total Return'].values, 1, 200)

def NN_model():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=21, use_bias=False))
    model.add(Dense(units=128, activation='relu', input_dim=21))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001) )
    return model


def get_update_operation(model):
    policy = model.output
    R = tf.placeholder('float')
    loss = tf.multiply(tf.log(policy), R)
    operation = tf.train.AdamOptimizer().minimize(loss)
    return operation, R

def policy_batch_update(model, update, r_holder, X, R, batch_size, iterations):
    K.set_learning_phase(1) #set learning phase
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(iterations):
            indices = np.random.choice(len(X), size=batch_size,replace = False )
            X_batch, R_batch = X[indices], np.reshape(R[indices],(batch_size, 1))
            sess.run([update], feed_dict={model.input:X_batch, r_holder:R_batch})
            print('Iteration: {}'.format(iteration))
    K.set_learning_phase(0) #set learning phase

model = NN_model()
update, r_holder = get_update_operation(model)
policy_batch_update(model, update, r_holder, X, R, 5, 100)

from agent.MarkovDecisionProcess import MDP

MDP.evaluate_maze_model(model=model, lookback=0, policy_type='softmax', method='policy-network')