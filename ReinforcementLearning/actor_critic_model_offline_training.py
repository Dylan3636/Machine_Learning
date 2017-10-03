# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:34:07 2017

@author: dylan
"""

import random
#setting seed
random.seed(5)
import pandas as pd
import numpy as np
#Initialize constants
LENGTH_OF_MAZE = 3
NUM_COLOURS = 1
NUM_ACTIONS = 3
ORIENTATION_DIM = 4
STATE_DIM = LENGTH_OF_MAZE**2 + ORIENTATION_DIM
SENSOR_DIM = 8
POLICY = 'softmax'
POLICY_LEARNING_ALGO = 'Q-learning'
TARGET_MODEL = 1
VANILLA=0
BATCH_SIZE = 5
ITERATIONS = 500
TAU = 1e-3
LR = 1e-3
NUM_STEPS = 50
NUM_EPISODES = 20
SEED=15,485,863
RANDOM_STATE = np.random.RandomState(seed=SEED)
DEBUG = 0
DISPLAY = 0
print(ITERATIONS)

# Preprocessing functions
def clean_state(state):
    tmp = state.split()[1::]
    tmp[-1] = state[-3] +'.'
    tmp= np.array(tmp, dtype=float)
    reading = tmp[-3::]
    one_hot = np.zeros(SENSOR_DIM)
    index = int(reading[0] + 2*reading[1] + 4*reading[2])
    one_hot[index] = 1
    return np.array(list(tmp[0:-3]) + list(one_hot), dtype=int)
    
def reading_encoder(reading):
    one_hot = np.zeros(SENSOR_DIM)
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
    encoded_action = np.zeros(NUM_ACTIONS)
    encoded_action[action_index]=1
    return encoded_action


def state_formatter(full_state):
    state = np.reshape(np.append(full_state[0], full_state[1]), [-1, STATE_DIM])
    readings = np.reshape(reading_encoder(full_state[2]), [-1, SENSOR_DIM])
    return [state, readings]


def vanilla_state_formatter(state):
    return np.reshape(state, [-1, 21])


def to_vanilla_state_formatter(state):
    return np.reshape(np.append(np.append(state[0], state[1]), reading_encoder(state[2])), [-1,21])


# Preprocessing
data = pd.read_csv('log_data_episode_5000.csv')
#dic = {100: 1, -10: -0.2, -1: -0.1 }
#data['Reward'] = data['Reward'].apply(lambda x: dic[x])
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

# Neural Network functions
def get_actor_model():
    state = Input(shape=[STATE_DIM])
    readings = Input(shape=[SENSOR_DIM])
    h1 = (Dense(units=64, activation='relu', input_dim=21, use_bias=False))(state)
    full_state = concatenate([h1, readings])
    h2 = Dense(units=128, activation='relu', input_dim=21)(full_state)
    out = Dense(units=NUM_ACTIONS, activation='softmax')(h2)
    model = Model(inputs=[state, readings], outputs=out)
    model.compile(loss='mse', optimizer=Adam(lr=1e-3))
    return model


def get_critic_action_model():
    state = Input(shape=[STATE_DIM])
    reading = Input(shape=[SENSOR_DIM])
    action = Input(shape=[NUM_ACTIONS])

    h1 = (Dense(units=64, activation='relu', use_bias=False))(state)
    full_state = concatenate([h1, reading])

    h2 = Dense(units=128, activation='relu')(full_state)
    full_state_with_actions = concatenate([h2, action])

    h3 = Dense(units=256, activation='relu')(full_state_with_actions)
    out = Dense(units=1, activation='linear')(h3)

    model = Model(inputs=[state, reading, action], outputs=out)
    model.compile(loss='mse', optimizer=Adam(lr=1e-3))
    return model


def get_vanilla_actor_model():
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=[21], use_bias=False))
    model.add(Dropout(0.9))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.9))
    model.add(Dense(units=NUM_ACTIONS, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    return model


def get_critic_model():
    state = Input(shape=[STATE_DIM])
    readings = Input(shape=[SENSOR_DIM])
    h1 = (Dense(units=64, activation='relu', input_dim=21, use_bias=False))(state)
    full_state = concatenate([h1, readings])
    h2 = Dense(units=128, activation='relu', input_dim=21)(full_state)
    out = Dense(units=NUM_ACTIONS, activation='linear')(h2)
    model = Model(inputs=[state, readings], outputs=out)
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model


def get_actor_update_operation(actor_model):
    policy = actor_model.output

    action_gradients = tf.placeholder('float', shape = [None, NUM_ACTIONS])

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


def update_target_models(models):
    actor_model, critic_model, actor_target_model, critic_target_model = models[0], models[1], models[2], models[3]

    # Training actor target model
    weights = actor_model.get_weights()
    target_weights = actor_target_model.get_weights()
    for i in range(len(weights)):
        target_weights[i] = TAU*weights[i] + (1-TAU)*target_weights[i]
    actor_target_model.set_weights(target_weights)

    # Training critic target model
    weights = critic_model.get_weights()
    target_weights = critic_target_model.get_weights()
    for i in range(len(weights)):
        target_weights[i] = TAU * weights[i] + (1 - TAU) * target_weights[i]
    critic_target_model.set_weights(target_weights)


def train_actor_critic_model(sess, models, episodes, gamma, tf_holders, iterations, batch_size, vanilla_actor):
    K.set_session(sess)
    actor_model = models[0]
    critic_model = models[1]
    np.random.seed(SEED)
    for iteration in range(iterations):
        if DEBUG:
            print('Iteration: {}'.format(iteration))
        else:
            print('\r','Iteration: {}'.format(iteration), end="")
        targets = []
        states = []
        readings = []
        actions = []
        deltas = []
        for _, frame in episodes.sample(batch_size, random_state = RANDOM_STATE).iterrows():
            state, action,reward, next_state, _, _, _, _ = frame
            state, reading, action = np.reshape(state[0:STATE_DIM], (1,STATE_DIM)), np.reshape(state[STATE_DIM::], (1,SENSOR_DIM)), np.reshape(action_encoder(action), (1,NUM_ACTIONS))
            next_state, next_reading = np.reshape(next_state[0:STATE_DIM],(1,STATE_DIM)), np.reshape(next_state[STATE_DIM::],(1,SENSOR_DIM))

            if TARGET_MODEL:
                q = models[3].predict([state, reading, action], batch_size=1).flatten()
            else:
                q = critic_model.predict([state, reading, action], batch_size=1).flatten()

            if not vanilla_actor:
                if TARGET_MODEL:
                    next_policy = models[2].predict([next_state, next_reading], batch_size=1).flatten()
                else:
                    next_policy = actor_model.predict([next_state, next_reading], batch_size=1).flatten()
            else:
                if TARGET_MODEL:
                    next_policy = models[2].predict(np.append(next_state, next_reading, axis=1), batch_size=1).flatten()
                else:
                    next_policy = actor_model.predict(np.append(next_state, next_reading, axis=1), batch_size=1).flatten()

            if reward == 1:
                target = reward
            else:
                # Using SARSA
                if POLICY_LEARNING_ALGO == 'SARSA':

                    # Epsilon-Greedy Policy
                    if POLICY == 'epsilon-greedy':
                        indx = np.argmax(next_policy)
                        if RANDOM_STATE.rand() < 0.5:
                            next_action = action_encoder(RANDOM_STATE.choice(NUM_ACTIONS))
                        else:
                            next_action = action_encoder(indx)

                    # Softmax policy
                    elif POLICY == 'softmax':
                        if not vanilla_actor:
                            next_action = action_encoder(RANDOM_STATE.choice(NUM_ACTIONS, p=next_policy))
                        else:
                            next_action = action_encoder(RANDOM_STATE.choice(NUM_ACTIONS, p=next_policy))

                elif POLICY_LEARNING_ALGO == 'Q-learning':
                    indx = np.argmax(next_policy)
                    next_action = action_encoder(indx)

                next_action = np.reshape(next_action, (1, NUM_ACTIONS))
                if TARGET_MODEL:
                    future_q = models[3].predict([next_state, next_reading, next_action], batch_size=1).flatten()
                else:
                    future_q = critic_model.predict([next_state, next_reading, next_action], batch_size=1).flatten()
                target = reward + gamma * future_q

            actions.append(action)
            deltas.append(target-q)
            targets.append(target)
            states.append(state)
            readings.append(reading)

        critic_model.train_on_batch([np.array(states).squeeze(axis=1), np.array(readings).squeeze(axis=1), np.array(actions).squeeze(axis=1)], np.array(targets))
        gradients = get_critic_gradients(sess, tf_holders[2], critic_model, np.array(states).squeeze(axis=1), np.array(readings).squeeze(axis=1), np.array(actions).squeeze(axis=1))
        gradients = np.squeeze(gradients)
        gradients = np.reshape(gradients,(-1, NUM_ACTIONS))

        if vanilla_actor:
            full_states=[np.append(state_reading[0], state_reading[1], axis=1) for state_reading in zip(states, readings)]
            sess.run([tf_holders[1]], feed_dict={actor_model.input: np.reshape(full_states, (batch_size, 21)), tf_holders[0]: gradients})
        else:
            sess.run([tf_holders[1]], feed_dict={actor_model.input[0]: np.array(states).squeeze(axis=1), actor_model.input[1]: np.array(readings).squeeze(axis=1), tf_holders[0]: gradients})

        if TARGET_MODEL:
            update_target_models(models)

        if DEBUG:
            print([state, reading, action])
            print(critic_model.predict([state, reading, action], batch_size=1).flatten())
            if vanilla_actor:
                print(actor_model.predict(np.append(state, reading, axis=1), batch_size=1).flatten())
            else:
                print(actor_model.predict([state, reading], batch_size=1).flatten())

sess = tf.Session()
K.set_learning_phase(1)  # set learning phase

actor_model = get_vanilla_actor_model() if VANILLA else get_actor_model()
critic_model = get_critic_action_model()
if TARGET_MODEL:
    target_actor_model = get_vanilla_actor_model() if VANILLA else get_actor_model()
    target_critic_model = get_critic_action_model()
else:
    target_actor_model=None
    target_critic_model=None

update_op, action_gradient_holder = get_actor_update_operation(actor_model)
gradient_op = get_gradient_operation(critic_model)
sess.run(tf.global_variables_initializer())
train_actor_critic_model(sess, [actor_model, critic_model, target_actor_model, target_critic_model], data, 0.75, [action_gradient_holder,update_op , gradient_op], ITERATIONS, BATCH_SIZE, VANILLA)


import matplotlib
if not DISPLAY:
    matplotlib.use('Agg')
from MarkovDecisionProcess import MDP
mdp = MDP(LENGTH_OF_MAZE**2, NUM_ACTIONS, state_formatter=to_vanilla_state_formatter if VANILLA else state_formatter, method='policy-network', policy=POLICY, q_model=actor_model)
from random_maze_environment import random_maze
env = random_maze(LENGTH_OF_MAZE, NUM_COLOURS, randomize_maze=1, randomize_state=0, random_state=RANDOM_STATE)
mdp.evaluate_model_in_environment(env, NUM_EPISODES, NUM_STEPS, show_env=DISPLAY)
#MDP.evaluate_maze_model(model=actor_model, policy_type=POLICY, method='policy-network', complex_input=0,state_formatter=vanilla_state_formatter )
