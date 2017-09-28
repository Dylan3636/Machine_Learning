import gym
import numpy as np
import matplotlib.pyplot as plt
from MarkovDecisionProcess import MDP
from sklearn.preprocessing import MinMaxScaler
import time
from collections import deque
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.optimizers import Adam


plt.ion()

def draw_graph(returns, ax):
    ax.cla()
    returns = np.array(returns)
    ave_returns = [np.mean((returns[i - 10:i])) for i in range(10, len(returns), 10)]
    ind = list(range(10, len(returns), 10))
    ax.plot(ind, ave_returns)
    ax.set_ylabel('Returns')
    ax.set_xlabel('Iteration')
    plt.pause(0.1)
    plt.show()

def solved(timesteps):
    timesteps = np.array(timesteps)
    return np.any(np.array([np.mean((timesteps[i - 100:i])) for i in range(99, len(timesteps), 1)]) > 195)

def build_model():
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_dim=4))
    model.add(Dropout(0.2))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=2, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.001) )
    return model

env_name ='CartPole-v0'
env = gym.make(env_name)
num_episodes = 5000

visualize = np.zeros(num_episodes)
for i in range(0, num_episodes, 50):
    visualize[i] =1
visualize_graphs = visualize
episodes = range(num_episodes)
timesteps = range(500)

num_states = env.observation_space.shape[0]
actions = range(env.action_space.n)

print('Action space: {}'.format(env.action_space))
print('Observation space: {}'.format(env.observation_space))

neurons = np.array([2**n for n in range(10)])
base_neuron = neurons[np.argmax((neurons-num_states)>=0)]
base_neuron = 256
mdp = MDP(num_actions=env.action_space.n, recurrent_layer=0, num_inputs=num_states, num_hidden_neurons=[base_neuron, 2 * base_neuron],
          num_hidden_layers=1, num_output_neurons=env.action_space.n, epsilon=1.0, gamma=0.99, epsilon_decay=0.9954, dropout_rate=0.2)

scaler = MinMaxScaler()
scaler.fit([env.observation_space.high, env.observation_space.low])

fig = plt.figure(1)
returns = []
timesteps_list=deque(maxlen=100)

for episode in episodes:
    previous_observation = env.reset() # initial observation
    done = 0
    for t in timesteps:
        if visualize[episode]:
            env.render()
            time.sleep(0.05)
        if done:
            print('Episode {} finished after {} timesteps.(Average timesteps: {})'.format(episode, t, np.mean(timesteps_list)))
            break
        action = mdp.make_decision(np.reshape(previous_observation, (-1, num_states))) # pick action
        next_observation, reward, done, info = env.step(action) # take action and see results
        if done and timesteps != 199:
            reward = -1000
        #next_observation = scaler.transform(np.reshape(next_observation, (1, num_states)))
        if np.random.rand()>0.2 or done: #Dropout
            mdp.update(np.reshape(previous_observation, (-1, num_states)), action, reward, np.reshape(next_observation, (-1, num_states)), done)

        previous_observation = next_observation
    returns.append(mdp.current_return)
    mdp.current_return=0
    timesteps_list.append(t)

    if visualize_graphs[episode]:
        ax = fig.add_subplot(1,1,1)
        draw_graph(returns=returns, ax=ax)

    if np.mean(timesteps_list) >= 195 and len(list(timesteps_list)) == 100:
        break

mdp.model.save('qnn_cart_model_1.h5')



