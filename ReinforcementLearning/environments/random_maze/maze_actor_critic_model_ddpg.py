# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:34:07 2017

@author: dylan
"""

#setting seed
import os
import sys
sys.path.append(os.path.abspath('..//..//'))
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
SEED = 5  # 15,485,863
np.random.seed(SEED)
import random
random.seed(SEED)
RANDOM_STATE = np.random.RandomState(seed=SEED)
import pandas as pd

#Initialize constants
LENGTH_OF_MAZE = 5
NUM_COLOURS = 1
ACTION_DIM = 3
ORIENTATION_DIM = 4
STATE_DIM = LENGTH_OF_MAZE**2 + ORIENTATION_DIM
SENSOR_DIM = 8
STOCHASTIC = 1
POLICY = 'softmax'
POLICY_LEARNING_ALGO = 'Q-learning'
TARGET_MODEL = 1
VANILLA=0
TRAIN = 1
BATCH_SIZE = 5
ITERATIONS = 0 if LENGTH_OF_MAZE == 3 else 0 # Data stored only for 3x3 maze. (Best for 3x3 maze: 40000/42000 iterations)
TAU = 1e-3
LR = 1e-3
HEAT = 0.2
HEAT_DECAY = 0.99985
MINIMUM_HEAT = 0.01
BETA = 1e-3
NUM_STEPS =400
NUM_EPISODES = 50000

LOAD_MODEL = 0
DEBUG = 0
DISPLAY = 1

import matplotlib
if not DISPLAY:
    matplotlib.use('Agg')


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
    encoded_action = np.zeros(ACTION_DIM)
    encoded_action[action_index] = 1
    return encoded_action


def state_formatter(full_state, action=None):
    state = np.reshape(np.append(full_state[0], full_state[1]), [-1, STATE_DIM])
    readings = np.reshape(reading_encoder(full_state[2]), [-1, SENSOR_DIM])
    if action is None:
        if VANILLA:
            np.append(state, readings, axis=0)
        else:
            return [state, readings]
    else:
        try:
            action = np.reshape(action_encoder(int(action)), [-1, ACTION_DIM])
        except:
            action = np.reshape(action, [-1, ACTION_DIM])
        return [state, readings, action]


def batch_state_formatter(full_states, actions=None):
    states = []
    readings = []
    if actions is None:
        for i, state in enumerate(full_states):
            state, reading = (state_formatter(state))
            states.append(state)
            readings.append(reading)
        return [np.reshape(states, [-1, STATE_DIM]), np.reshape(readings, [-1, SENSOR_DIM])]
    else:
        action_list = []
        for i, state in enumerate(full_states):
            state, reading, action = (state_formatter(state, actions[i]))
            states.append(state)
            readings.append(reading)
            action_list.append(action)
        return [np.reshape(states, [-1, STATE_DIM]), np.reshape(readings, [-1, SENSOR_DIM]), np.reshape(action_list, [-1, ACTION_DIM])]


def vanilla_state_formatter(state):
    return np.reshape(state, [-1, 21])


def to_vanilla_state_formatter(state):
    return np.reshape(np.append(np.append(state[0], state[1]), reading_encoder(state[2])), [-1,21])


# Preprocessing
data = pd.read_csv('log_data_episode_5000.csv')
data['Previous State'] = data['Previous State'].apply(clean_state).values
data['Current State'] = data['Current State'].apply(clean_state).values

# Neural Network imports
import tensorflow as tf
tf.set_random_seed(SEED)
from keras.models import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import concatenate
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras import backend as K


# Neural Network functions
def entropy(policy):
    return -tf.reduce_sum(policy*tf.nn.log_softmax(policy))

def get_actor_model():
    state = Input(shape=[STATE_DIM])
    readings = Input(shape=[SENSOR_DIM])
    h1 = (Dense(units=64, activation='relu', input_dim=21, use_bias=False))(state)
    full_state = concatenate([h1, readings])
    h2 = Dense(units=128, activation='relu', input_dim=21)(full_state)
    out = Dense(units=ACTION_DIM, activation='softmax')(h2)
    model = Model(inputs=[state, readings], outputs=out)
    model.compile(loss='mse', optimizer=Adam(lr=1e-3))
    return model


def get_critic_action_model():
    state = Input(shape=[STATE_DIM])
    reading = Input(shape=[SENSOR_DIM])
    action = Input(shape=[ACTION_DIM])

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
    model.add(Dense(units=ACTION_DIM, activation='softmax'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    return model


def get_critic_model():
    state = Input(shape=[STATE_DIM])
    readings = Input(shape=[SENSOR_DIM])
    h1 = (Dense(units=64, activation='relu', input_dim=21, use_bias=False))(state)
    full_state = concatenate([h1, readings])
    h2 = Dense(units=128, activation='relu', input_dim=21)(full_state)
    out = Dense(units=ACTION_DIM, activation='linear')(h2)
    model = Model(inputs=[state, readings], outputs=out)
    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model


def get_actor_update_operation(actor_model):
    policy = actor_model.output
    weights = actor_model.trainable_weights

    critic_gradients = tf.placeholder('float', shape=[None, ACTION_DIM])
    loss = -tf.nn.log_softmax(policy)*critic_gradients - BETA*entropy(policy)
    gradients = tf.gradients(ys=loss, xs=weights)
    grads = zip(gradients, weights)
    operation = tf.train.AdamOptimizer(LR).apply_gradients(grads)

    return operation, critic_gradients


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
    actor_model = models[0]
    critic_model = models[1]
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
            state, reading, action = np.reshape(state[0:STATE_DIM], (1,STATE_DIM)), np.reshape(state[STATE_DIM::], (1,SENSOR_DIM)), np.reshape(action_encoder(action), (1, ACTION_DIM))
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
                if not STOCHASTIC:
                    # Using SARSA
                    if POLICY_LEARNING_ALGO == 'SARSA':

                        # Epsilon-Greedy Policy
                        if POLICY == 'epsilon-greedy':
                            indx = np.argmax(next_policy)
                            if RANDOM_STATE.rand() < 0.5:
                                next_action = action_encoder(RANDOM_STATE.choice(ACTION_DIM))
                            else:
                                next_action = action_encoder(indx)

                        # Softmax policy
                        elif POLICY == 'softmax':
                            if not vanilla_actor:
                                next_action = action_encoder(RANDOM_STATE.choice(ACTION_DIM, p=next_policy))
                            else:
                                next_action = action_encoder(RANDOM_STATE.choice(ACTION_DIM, p=next_policy))

                    elif POLICY_LEARNING_ALGO == 'Q-learning':
                        indx = np.argmax(next_policy)
                        next_action = action_encoder(indx)
                else:
                    next_action = next_policy
                next_action = np.reshape(next_action, (1, ACTION_DIM))
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
        actions_for_grad = models[3].predict_on_batch([np.array(states).squeeze(axis=1), np.array(readings).squeeze(axis=1)])
        gradients = get_critic_gradients(sess, tf_holders[2], critic_model, np.array(states).squeeze(axis=1), np.array(readings).squeeze(axis=1), actions_for_grad)
        gradients = np.squeeze(gradients)
        gradients = np.reshape(gradients, (-1, ACTION_DIM))

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

K.set_learning_phase(1)  # set learning phase
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf)
K.set_session(sess)

from keras.models import load_model
if LOAD_MODEL:
    target_actor_model = load_model('actor_model_{}.h5'.format(ITERATIONS))
    target_critic_model = load_model('critic_model_{}.h5'.format(ITERATIONS))
    actor_model = load_model('actor_model_{}.h5'.format(ITERATIONS))
    critic_model = load_model('critic_model_{}.h5'.format(ITERATIONS))
    update_op, action_gradient_holder = get_actor_update_operation(actor_model)
    gradient_op = get_gradient_operation(critic_model)
else:
    actor_model = get_vanilla_actor_model() if VANILLA else get_actor_model()
    critic_model = get_critic_action_model()
    if TARGET_MODEL:
        target_actor_model = get_vanilla_actor_model() if VANILLA else get_actor_model()
        target_critic_model = get_critic_action_model()
    else:
        target_actor_model=None
        target_critic_model=None

    update_op, action_gradient_holder = get_actor_update_operation(actor_model)
    gradient_op = get_gradient_operation(target_critic_model)
    sess.run(tf.global_variables_initializer())
    train_actor_critic_model(sess, [actor_model, critic_model, target_actor_model, target_critic_model], data, 0.75, [action_gradient_holder,update_op , gradient_op], ITERATIONS, BATCH_SIZE, VANILLA)

from agents.MarkovDecisionProcess import MDP
from environments.tools import evaluate_agent_in_environment
mdp = MDP(LENGTH_OF_MAZE ** 2, ACTION_DIM, state_formatter=to_vanilla_state_formatter if VANILLA else state_formatter, method='actor-critic',
          policy_type=POLICY, actor_model=actor_model, critic_model=critic_model, target_models=[target_actor_model, target_critic_model], sess=sess,
          random_state=RANDOM_STATE, heat=HEAT, heat_decay=HEAT_DECAY, minimum_heat=MINIMUM_HEAT, stochastic=1)
mdp.ac_toolkit.set_formatters(state_formatter=state_formatter, batch_state_formatter=batch_state_formatter)
mdp.ac_toolkit.set_actor_update_op(actor_update_op=update_op, critic_gradient_holder=action_gradient_holder)
mdp.ac_toolkit.set_critic_gradient_operation(critic_gradient_op=gradient_op)

from environments.random_maze.random_maze_environment import random_maze
env = random_maze(LENGTH_OF_MAZE, NUM_COLOURS, randomize_maze=1, randomize_state=1, random_state=RANDOM_STATE)
evaluate_agent_in_environment(mdp, env, NUM_EPISODES, NUM_STEPS, show_env=list(range(0, NUM_EPISODES, 100)), train=TRAIN)

# Saving models
mdp.actor_model.save('actor_model_{}_maze_length_{}.h5'.format(ITERATIONS, LENGTH_OF_MAZE))
mdp.critic_model.save('critic_model_{}_maze_length_{}.h5'.format(ITERATIONS, LENGTH_OF_MAZE))

#MDP.evaluate_maze_model(model=actor_model, policy_type=POLICY, method='policy-network', complex_input=0,state_formatter=vanilla_state_formatter )
