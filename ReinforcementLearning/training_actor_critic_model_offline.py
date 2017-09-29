# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:34:07 2017

@author: dylan
"""


import pandas as pd
import numpy as np
DEBUG = 1
POLICY = 'softmax'
BATCH_SIZE = 5
ITERATIONS = 200
LR = .5e-3
NUM_STEPS = 50
NUM_EPISODES = 100
DISPLAY=0
def clean_state(state):
    tmp = state.split()[1::]
    tmp[-1] = state[-3] +'.'
    tmp= np.array(tmp, dtype=float)
    reading = tmp[-3::]
    one_hot = np.zeros(8)
    index = int(reading[0] + 2*reading[1] + 4*reading[2])
    one_hot[index] = 1
    return np.array(list(tmp[0:-3]) + list(one_hot), dtype=int)
    
def reading_encoder(reading):
    one_hot = np.zeros(8)
    index = int(reading[0] + 2*reading[1] + 4*reading[2])
    one_hot[index] = 1
    return one_hot
 
def sum_rewards(episodes):
    episodes = episodes.copy()
    episodes['Return'] = None
    
    for name, episode in episodes.groupby('Episode'):
        print(name)
        rows = episode.index.values
        episodes.set_value(rows, 'Total Return', sum(episode['Reward'])*np.ones(len(episode)) )
    return episodes

def action_encoder(action_index):
    encoded_action = np.zeros(3)
    encoded_action[action_index]=1
    return encoded_action
    
    
def state_formatter(full_state):
    state = np.reshape(full_state[0:13], [-1, 13])
    readings = np.reshape(full_state[13::], [-1, 8])
    return [state, readings]

def vanilla_state_formatter(state):
    return np.reshape(state, [-1, 21])

def to_vanilla_state_formatter(state):
    return np.reshape(np.append(np.append(state[0], state[1]), reading_encoder(state[2])), [-1,21])



# Preprocessing
data = pd.read_csv('log_data_episode_5000.csv')
#dic = {100: 1, -10: -0.2, -1: -0.1 }
#data['Reward'] = data['Reward'].apply(lambda x: dic[x])
#del(tmp)
data['Previous State'] = data['Previous State'].apply(clean_state).values
data['Current State'] = data['Current State'].apply(clean_state).values

# Neural Network imports
from keras.models import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
import tensorflow as tf
from keras import backend as K


def actor_model():
    state = Input(shape=[13])
    readings = Input(shape=[8])
    h1 = (Dense(units=64, activation='relu', input_dim=21, use_bias=False))(state)
    full_state = concatenate([h1, readings])
    h2 = Dense(units=128, activation='relu', input_dim=21)(full_state)
    out = Dense(units=3, activation='softmax')(h2)
    model = Model(inputs=[state, readings], outputs=out)
    model.compile(loss='mse', optimizer=Adam(lr=1e-3))
    return model

def critic_action_model():
    state = Input(shape=[13])
    reading = Input(shape=[8])
    action = Input(shape=[3])

    h1 = (Dense(units=64, activation='relu', use_bias=False))(state)
    full_state = concatenate([h1, reading])

    h2 = Dense(units=128, activation='relu')(full_state)
    full_state_with_actions = concatenate([h2, action])

    h3 = Dense(units=256, activation='relu')(full_state_with_actions)
    out = Dense(units=1, activation='linear')(h3)

    model = Model(inputs=[state, reading, action], outputs=out)
    model.compile(loss='mse', optimizer=Adam(lr=1e-3))
    return model

def actor_model_vanilla():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=[21], use_bias=False))
    model.add(Dropout(0.9))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.9))
    model.add(Dense(units=3, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    return model


def critic_model():
    state = Input(shape=[13])
    readings = Input(shape=[8])
    h1 = (Dense(units=64, activation='relu', input_dim=21, use_bias=False))(state)
    full_state = concatenate([h1, readings])
    h2 = Dense(units=128, activation='relu', input_dim=21)(full_state)
    out = Dense(units=3, activation='linear')(h2)
    model = Model(inputs=[state, readings], outputs=out)
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model



def get_actor_update_operation(actor_model):
    policy = actor_model.output

    action_gradients = tf.placeholder('float', shape = [None, 3])
    #loss = tf.nn.log_softmax(policy)

    weights = actor_model.trainable_weights
    gradient_parameters = tf.gradients(policy, weights, -action_gradients)
    grads = zip(gradient_parameters, weights)

    operation = tf.train.AdamOptimizer(LR).apply_gradients(grads)
    return operation, action_gradients

def get_gradient_operation(critic_model):
    Q_function = critic_model.output
    actions = critic_model.inputs[2]
    action_gradient_op = tf.gradients(Q_function, actions)
    return action_gradient_op

def get_critic_gradients(sess, gradient_op, critic_model, state, reading, action):
    return sess.run([gradient_op],feed_dict = {critic_model.inputs[0] : state, critic_model.inputs[1] : reading, critic_model.inputs[2] : action, })


def train_actor_critic_model(sess, actor_model, critic_model, episodes, gamma,actor_update_operation, holders, iterations, batch_size, actor_vanilla):
    K.set_session(sess)
    for iteration in range(iterations):
        print('Iteration: {}'.format(iteration))
        targets = []
        states = []
        readings = []
        actions = []
        deltas = []
        for _, frame in episodes.sample(batch_size).iterrows():
            state, action,reward, next_state,_,_,_,_ = frame
            state, reading, action = np.reshape(state[0:13], (1,13)), np.reshape(state[13::], (1,8)), np.reshape(action_encoder(action), (1,3))
            next_state, next_reading = np.reshape(next_state[0:13],(1,13)), np.reshape(next_state[13::],(1,8))
            q = critic_model.predict([state, reading, action], batch_size=1).flatten()

            if not actor_vanilla:
                next_policy = actor_model.predict([next_state, next_reading], batch_size=1).flatten()
            else:
                next_policy = actor_model.predict(np.append(next_state, next_reading, axis=1), batch_size=1).flatten()

            if reward == 1:
                target = reward
            else:
                #Using SARSA

                #Epsilon-Greedy Policy
                if POLICY == 'epsilon-greedy':
                    indx = np.argmax(next_policy)
                    if np.random.rand() < 0.5:
                        next_action = action_encoder(np.random.choice(3))
                    else:
                        next_action = action_encoder(indx)

                #Softmax policy
                elif POLICY == 'softmax':

                    if not actor_vanilla:
                        next_action = action_encoder(np.random.choice(3, p=next_policy))
                    else:
                        next_action = action_encoder(np.random.choice(3, p=next_policy))
                
                next_action = np.reshape(next_action, (1, 3))
                future_q = critic_model.predict([next_state, next_reading, next_action], batch_size=1).flatten()

                target = reward + gamma * future_q
            actions.append(action)
            deltas.append(target-q)
            targets.append(target)
            states.append(state)
            readings.append(reading)

        critic_model.train_on_batch([np.array(states).squeeze(axis=1), np.array(readings).squeeze(axis=1), np.array(actions).squeeze(axis=1)], np.array(targets))
        gradients = get_critic_gradients(sess, holders[1], critic_model, np.array(states).squeeze(axis=1), np.array(readings).squeeze(axis=1), np.array(actions).squeeze(axis=1))
        gradients = np.squeeze(gradients)
        gradients = np.reshape(gradients,(-1,3))
        if actor_vanilla:
            full_states=[np.append(state_reading[0], state_reading[1], axis=1) for state_reading in zip(states, readings)]
            sess.run([actor_update_operation], feed_dict={actor_model.input: np.reshape(full_states, (batch_size, 21)), holders[0]: gradients})
        else:
            sess.run([actor_update_operation], feed_dict={actor_model.input[0]: np.array(states).squeeze(axis=1), actor_model.input[1]: np.array(readings).squeeze(axis=1), holders[0]: gradients})

        if DEBUG:
            print(critic_model.predict([state, reading, action], batch_size=1).flatten())
            if actor_vanilla:
                print(actor_model.predict(np.append(state, reading, axis=1), batch_size=1).flatten())
            else:
                print(actor_model.predict([state, reading], batch_size=1).flatten())

sess = tf.Session()
K.set_learning_phase(1)  # set learning phase
actor_model = actor_model_vanilla()
update, action_gradient_holder = get_actor_update_operation(actor_model)
sess.run(tf.global_variables_initializer())
critic_model = critic_action_model()
gradient_op = get_gradient_operation(critic_model)
train_actor_critic_model(sess, actor_model, critic_model, data, 0.75, update, [action_gradient_holder, gradient_op], ITERATIONS, BATCH_SIZE, True)


import matplotlib
if not DISPLAY:
    matplotlib.use('Agg')
from MarkovDecisionProcess import MDP
mdp = MDP(9,3,state_formatter=to_vanilla_state_formatter, method='policy-network', policy=POLICY, q_model=actor_model)
from random_maze_environment import random_maze
env = random_maze(3,1)
mdp.evaluate_model(env, NUM_EPISODES, NUM_STEPS)
#MDP.evaluate_maze_model(model=actor_model, policy_type=POLICY, method='policy-network', complex_input=0,state_formatter=vanilla_state_formatter )
