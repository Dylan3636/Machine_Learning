# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 21:35:16 2017

@author: dylan
"""
import pandas as pd
import numpy as np
from collections import deque
    
def group_states(episodes, lookback, num_features):
    episodes = episodes.copy()
    episodes['Grouped State'] = None
    
    for name, episode in episodes.groupby('Episode'):
        tmp = deque(np.zeros((lookback+1, num_features)), maxlen = lookback+1)
        print(name)
        rows_to_drop = []
        for name, step in episode.iterrows():
            if len(episode) >31:
                rows_to_drop.append(name)
            else:
                tmp.append(clean_state(step['Previous State']))
                episodes.set_value(name, 'Grouped State', np.reshape(tmp, (len(tmp), num_features)))
        episodes.drop(rows_to_drop, axis=0)
    return episodes


def action_encoder(action_index):
    tmp = np.zeros(3)
    tmp[action_index] = 1
    return np.array(tmp, dtype=int).reshape((3))

def clean_state(state):
    tmp = state.split()[1::]
    tmp[-1] = state[-3] +'.'
    tmp= np.array(tmp, dtype=float)
    orientation = tmp[-3::]
    one_hot = np.zeros(8)
    index = int(orientation[0] + 2*orientation[1] + 4*orientation[2])
    one_hot[index] = 1
    return np.array(list(tmp[0:-3]) + list(one_hot), dtype=int)


#Defining constants
NUM_FEATS = 21
LOOKBACK = 1


# Preprocessing
data = pd.read_csv('log_data_episode_30000')
tmp = data['Current State']
data['Current State'] = data['Reward']
dic = {100: 1, -10: -0.2, -1: -0.1 }
data['Reward'] = tmp.apply(lambda x: dic[x])
del(tmp)
data['Previous State'] = data['Previous State'].apply(clean_state).values

grouped_data = group_states(data, LOOKBACK, NUM_FEATS)
grouped_data = grouped_data.dropna()
X = list(grouped_data['Grouped State'])
y = list(grouped_data['Action Taken'].apply(action_encoder).values)


from keras.models import Sequential
from keras.layers import Input
from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.regularizers import l2


#model = RNN_Model((lookback+1, num_features))
#model.fit(np.reshape(X,(-1,lookback+1,num_features)), np.reshape(y,(-1, 3)), batch_size = 4000, shuffle=True, nb_epoch=50)
#model = NN_model()
#model.fit(np.array(list(X)), np.array(list(y)), batch_size = 500, shuffle=True, epochs=50)
#model = train_Q_model(model, successful_episodes, 0.75, 10000, 1  )


