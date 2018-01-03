# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 21:44:53 2017

@author: dylan
"""

import pandas as pd
import numpy as np


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
data = pd.read_csv('log_data_episode_30000')
tmp = data['Current State']
data['Current State'] = data['Reward']
dic = {100: 1, -10: -0.2, -1: -0.1 }
data['Reward'] = tmp.apply(lambda x: dic[x])
del(tmp)
data['Previous State'] = data['Previous State'].apply(clean_state).values
data = data.groupby('Episode').apply(sum_rewards)


# Neural Network imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam


def NN_model():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=21, use_bias=False))
    model.add(Dense(units=128, activation='relu', input_dim=21))
    model.add(Dropout(0.8))
    model.add(Dense(units=3, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001) )
    return model


def train_Q_model(model, episodes, gamma, iterations, batch_size):  
    for iteration in range(iterations):
        print('Iteration: {}'.format(iteration))
        targets = []
        states = []  
        for _, frame in episodes.sample(batch_size).iterrows():
            _, state, action, next_state, reward,_,_,_ = frame
            state = clean_state(state)
            next_state = clean_state(next_state)
            q = model.predict(np.reshape(state, (-1, len(state))), batch_size=1).flatten()
            if reward:
                target = reward
            else:
                future_q = np.max(model.predict(np.reshape(next_state, (-1, len(state))), batch_size=1).flatten())
                target = reward + gamma * future_q
            q[action] = target
            #q = np.reshape(q, (-1, len(q)))
            targets.append(q)
            states.append(state)
        model.fit(np.array(states), np.array(targets), batch_size=batch_size, epochs=1)

model = NN_model()
train_Q_model(model, data, gamma=0.7, iterations=10000, batch_size=5)