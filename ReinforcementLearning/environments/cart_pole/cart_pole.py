import os
import sys

import gym
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath('../'))
from agents.MarkovDecisionProcess import MDP
from environments.tools import evaluate_model_in_environment
from sklearn.preprocessing import MinMaxScaler
import time
import keras.backend as K
from collections import deque
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import concatenate
from keras.optimizers import Adam

plt.ion()
def draw_graph(returns, ax):
    ax.cla()
    returns = np.array(returns)
    ave_returns = [np.mean((returns[i - 10:i])) for i in range(10, len(returns), 1)]
    ind = list(range(10, len(returns), 1))
    ax.plot(ind, ave_returns)
    ax.set_ylabel('Returns')
    ax.set_xlabel('Iteration')
    plt.pause(0.1)
    plt.show()

def solved(timesteps):
    timesteps = np.array(timesteps)
    return np.any(np.array([np.mean((timesteps[i - 100:i])) for i in range(99, len(timesteps), 1)]) > 195)

def rbf(x, c, h):
    return K.exp(-K.abs(x-c)/h**2)

def build_rbf_model():
    theta = Input(shape=[1])
    theta_rbf_1 = Lambda(function=rbf, arguments={'c':-0.2, 'h':0.4})(theta)
    theta_rbf_2 = Lambda(function=rbf, arguments={'c':0.2, 'h':0.4})(theta)
    theta_dot = Input(shape=[1])
    thetas = concatenate([theta_rbf_1, theta_rbf_2, theta_dot])
    thetas_h = Dense(24, activation='tanh')(thetas)

    x = Input(shape=[1])
    x_rbf_1 = Lambda(function=rbf, arguments={'c': -3, 'h': 2})(x)
    x_rbf_2 = Lambda(function=rbf, arguments={'c': -1, 'h': 2})(x)
    x_rbf_3 = Lambda(function=rbf, arguments={'c': 1, 'h': 2})(x)
    x_rbf_4 = Lambda(function=rbf, arguments={'c': 3, 'h': 2})(x)
    x_dot = Input(shape=[1])
    xs = concatenate([x_rbf_1, x_rbf_2, x_rbf_3, x_rbf_4, x_dot])
    xs_h = Dense(24, activation='tanh')(xs)

    joined = concatenate([xs_h, thetas_h])

    hidden_2 = Dense(128, activation='tanh')(joined)
    out = Dense(2, activation='linear')(hidden_2)
    model = Model(inputs=[x, x_dot, theta, theta_dot], outputs=[out])
    model.compile(loss='mse', optimizer=Adam(lr=0.01, decay=0.01))
    return model

def rbf_state_formatter(observations):
    observations = np.squeeze(observations)
    return [np.array([observations[0]]), np.array([observations[1]]), np.array([observations[2]]), np.array([observations[3]])]


def build_model():

    model = Sequential()

    model.add(Dense(units=24, activation='tanh', input_dim=4))
    #model.add(Dropout(0.2))
    model.add(Dense(units=48, activation='tanh'))
    #model.add(Dropout(0.2))
    model.add(Dense(units=2, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.01, decay=0.01) )
    return model

env_name ='CartPole-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
actions = range(env.action_space.n)

def state_formatter(observations):
    return np.reshape(observations, (-1, state_dim))

num_episodes = 5000
episodes = range(num_episodes)
timesteps = range(500)
visualize = np.zeros(num_episodes)
for i in range(0, num_episodes, 50):
    visualize[i] =1
visualize_graphs = visualize

print('Action space: {}'.format(env.action_space))
print('Observation space: {}'.format(env.observation_space))

mdp = MDP(num_actions=env.action_space.n, state_formatter=rbf_state_formatter,q_model=build_rbf_model(), epsilon=1.0, gamma=0.99, epsilon_decay=0.9954, minimum_epsilon=0, dropout_rate=0.2)

def reward_formatter(observation):
    _, reward, done, t = observation
    if done and t != 199:
        reward = 0
    else:
        reward /= 10.0
    return reward


evaluate_model_in_environment(mdp, env, num_episodes, num_timesteps=400, show_env=list(visualize), train=1, reward_formatter=reward_formatter, delay=0.05, is_converged=lambda x: x >= 19.5)


# scaler = MinMaxScaler()
# scaler.fit([env.observation_space.high, env.observation_space.low])
#
# fig = plt.figure(1)
# returns = []
# timesteps_list=deque(maxlen=100)
#
# for episode in episodes:
#     previous_observation = env.reset() # initial observation
#     done = 0
#     for t in timesteps:
#         if visualize[episode]:
#             env.render()
#             time.sleep(0.05)
#         action = mdp.make_decision(np.reshape(previous_observation, (-1, state_dim))) # pick action
#         next_observation, reward, done, info = env.step(action) # take action and see results
#         if done and t!= 199:
#             reward = 0
#         else:
#             reward /= 10.0
#         if episode > 0:
#             mdp.update(previous_observation, action, reward, next_observation, done, replay=0.8, dropout=0, log=True,
#                       batch_size=1, minibatch_size=1)
#         else:
#             mdp.update(previous_observation, action, reward, next_observation, done, dropout=1, log=True)
#
#         if done:
#             print('Episode {} finished after {} timesteps.(Average timesteps: {})'.format(episode, t,
#                                                                                           np.mean(timesteps_list)))
#             break
#
#         previous_observation = next_observation
#     returns.append(mdp.current_return)
#     mdp.current_return=0
#     timesteps_list.append(t)
#
#     if visualize_graphs[episode]:
#         ax = fig.add_subplot(1,1,1)
#         draw_graph(returns=returns, ax=ax)
#
#     if np.mean(timesteps_list) >= 195 and len(list(timesteps_list)) == 100:
#         break
#
# mdp.model.save('qnn_cart_model_1.h5')



