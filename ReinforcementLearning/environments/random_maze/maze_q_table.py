# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:34:07 2017

@author: dylan
"""

# setting seed
import os
import sys
from itertools import product

sys.path.append(os.path.abspath('..//..//'))
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np

SEED = 5  # 15,485,863
np.random.seed(SEED)
import random

random.seed(SEED)
RANDOM_STATE = np.random.RandomState(seed=SEED)
import pandas as pd

# Initialize constants
# Maze parameters
LENGTH_OF_MAZE = 9
NUM_COLOURS = 1
ACTION_DIM = 3
ORIENTATION_DIM = 4
STATE_DIM = LENGTH_OF_MAZE ** 2 + ORIENTATION_DIM
SENSOR_DIM = 8


GAMMA = 0.9
LR = 1e-3
ITERATIONS = 0
POLICY = 'softmax'
HEAT=0.2
HEAT_DECAY = 1-1e-3
HEAT_MIN = 0.1
EPSILON = 0.4
EPSILON_DECAY = 1-1e-3
EPSILON_MIN = 0.1
POLICY_LEARNING_ALGO = 'Q-learning'

NUM_STEPS = 400
NUM_EPISODES = 50000

TRAIN = 1
LOAD_MODEL = 0
DEBUG = 0
DISPLAY = 1

import matplotlib

if not DISPLAY:
    matplotlib.use('Agg')


# Preprocessing functions
def clean_state(state):
    tmp = state.split()[1::]
    tmp[-1] = state[-3] + '.'
    tmp = np.array(tmp, dtype=float)
    reading = tmp[-3::]
    one_hot = np.zeros(SENSOR_DIM)
    index = int(reading[0] + 2 * reading[1] + 4 * reading[2])
    one_hot[index] = 1
    return np.array(list(tmp[0:-3]) + list(one_hot), dtype=int)


def reading_encoder(reading):
    if len(reading)<8:
        one_hot = np.zeros(SENSOR_DIM)
        index = int(reading[0] + 2 * reading[1] + 4 * reading[2])
        one_hot[index] = 1
        return one_hot
    else:
        return reading


def sum_rewards(episodes):
    episodes = episodes.copy()
    episodes['Return'] = None

    for name, episode in episodes.groupby('Episode'):
        print(name)
        rows = episode.index.values
        episodes.set_value(rows, 'Total Return', sum(episode['Reward']) * np.ones(len(episode)))
    return episodes


def action_encoder(action_index):
    encoded_action = np.zeros(ACTION_DIM)
    encoded_action[action_index] = 1
    return encoded_action


def get_state_index(full_state):
    """
    Formatter function to be used as the agent's state_formatter function. Takes the state and returns the corresponding index in Q-table
    :param full_state: list of maze environment observations
    :return state index:
    """
    position, orientation, readings = full_state
    d = dict(enumerate(product(range(STATE_DIM-ORIENTATION_DIM), range(ORIENTATION_DIM), range(SENSOR_DIM))))
    d = dict(zip(d.values(), d.keys()))
    position = np.argmax(position)
    orientation = np.argmax(orientation)
    readings = np.argmax(readings)
    return d[(position, orientation, readings)]


# Preprocessing
data = pd.read_csv('log_data_episode_5000.csv')
data['Previous State'] = data['Previous State'].apply(clean_state).values
data['Current State'] = data['Current State'].apply(clean_state).values


def train_q_table_offline(q_table, episodes, iterations):
    for _, frame in episodes.sample(frac=iterations, random_state=RANDOM_STATE).iterrows():
        state, action, reward, next_state = frame.iloc[:, 0], frame.iloc[:, 1], frame.iloc[:, 2], frame.iloc[:, 3]
        state_index = get_state_index(state)
        next_state_index = get_state_index(next_state)
        future_q_values = q_table[next_state_index]
        if POLICY_LEARNING_ALGO == 'Q-Learning':
            target = GAMMA*np.max(future_q_values)
        elif POLICY_LEARNING_ALGO == 'SARSA':
            if POLICY=='softmax':
                p = np.exp(future_q_values)/sum(np.exp(future_q_values))
                target = GAMMA*np.dot(future_q_values, p)
                HEAT = min(HEAT_MIN, HEAT*HEAT_DECAY)
            elif POLICY == 'epsilon-greedy':
                if RANDOM_STATE.rand()< EPSILON:
                    target = GAMMA*RANDOM_STATE.choice(future_q_values)
                else:
                    target = GAMMA*np.max(future_q_values)
                EPSILON = min(EPSILON_MIN, EPSILON*EPSILON_DECAY)
        q_table[state_index, action] += LR*(reward + target - q_table[state_index, action])
    return q_table


if LOAD_MODEL:
    q_table = np.matrix(np.load('q-table.txt'))
else:
    N = (STATE_DIM-SENSOR_DIM)*(SENSOR_DIM)*(ORIENTATION_DIM)
    q_table=np.matrix(np.zeros([N, ACTION_DIM]))
    train_q_table_offline(q_table=q_table, episodes=data, iterations=ITERATIONS)
from agents.MarkovDecisionProcess import MDP
from environments.tools import evaluate_model_in_environment

mdp = MDP(LENGTH_OF_MAZE ** 2, ACTION_DIM, state_formatter=get_state_index,init_Q_matrix=q_table, random_state=RANDOM_STATE, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, minimum_epsilon=EPSILON_MIN)

from environments.random_maze.random_maze_environment import random_maze

env = random_maze(LENGTH_OF_MAZE, NUM_COLOURS, randomize_maze=1, randomize_state=1, random_state=RANDOM_STATE)
evaluate_model_in_environment(mdp, env, NUM_EPISODES, NUM_STEPS, show_env=list(range(0, NUM_EPISODES, 100)),
                              train=TRAIN)
# Saving models
np.savetxt('q-table', mdp.Q_matrix)
# MDP.evaluate_maze_model(model=actor_model, policy_type=POLICY, method='policy-network', complex_input=0,state_formatter=vanilla_state_formatter )
