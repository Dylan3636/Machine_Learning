import sys
import os
import imageio
sys.path.insert(0, (('\\User\dylan\Documents\Github\KaSeDy\pybot')))
from StateEstimation.BayesFilter import DiscreteBayesFilter
from tools.Map import Map

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dropout

import numpy as np
from pandas import DataFrame
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
class POMDP:
    def __init__(self,num_states, num_actions, transition_model=None, sensor_model=None, initial_belief=None,q_model=None , v_model=None,network_type ='Q',recurrent=False, state_estimator='BF', immediate_rewards=None, gamma =0.9, epsilon =0.1, epsilon_decay=0.995, num_inputs=None, num_hidden_neurons=20, num_hidden_layers=1, lookback=1, num_rnn_units=64, num_output_neurons=4, sensor_readings=None, sensor_probabilities=None, actions=None, log=None):
        if state_estimator == 'BF':
            self.state_estimator = DiscreteBayesFilter(transition_model, sensor_model, initial_belief)
        self.immediate_rewards = immediate_rewards
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.num_states = num_states
        self.num_actions = num_actions
        if q_model is None and v_model is None:
            if recurrent:
                self.model = self._build_RNN(num_states if num_inputs is None else num_inputs, lookback=lookback, num_rnn_units=num_rnn_units, num_hidden_neurons=num_hidden_neurons, num_hidden_layers=num_hidden_layers, num_output_neurons=num_output_neurons) #Q-Neural Network
            else:
                self.model = self._build_NN(num_states if num_inputs is None else num_inputs, num_hidden_neurons, num_hidden_layers, num_output_neurons) #
        elif v_model is None:
            self.model = q_model
        else:
            self.model = v_model
        self.network_type = network_type.capitalize()
        self.sensor_readings = sensor_readings
        self.sensor_probabilities = sensor_probabilities
        self.actions = actions
        self.current_return = 0
        self.recurrent = recurrent
        if log is None:
            self.log = DataFrame(columns=['Previous State', 'Action Taken', 'Current State', 'Reward', 'Completed'])
        else:
            self.log = log


    def _build_NN(self, num_inputs,num_hidden_neurons, num_hidden_layers=1, num_output_neurons=4, dropout_rate=0.2):
        model = Sequential()
        if num_hidden_layers == 0:
            model.add(Dense(units=self.num_actions, input_dim=num_inputs, activation='linear'))
            model.add(Dropout(dropout_rate))
        else:
            if type(num_hidden_neurons) is int:
                num_hidden_neurons = np.asarray(num_hidden_neurons*np.ones(num_hidden_layers), dtype=int)
            else:
                num_hidden_layers = len(num_hidden_neurons)
            model.add(Dense(units= num_hidden_neurons[0], input_dim=num_inputs, activation='relu'))
            model.add(Dropout(dropout_rate))
            for layer in range(0,num_hidden_layers-1):
                model.add(Dense(units=num_hidden_neurons[layer], activation='relu'))
                model.add(Dropout(dropout_rate))
            model.add(Dense(units = num_output_neurons, activation='linear'))
            model.add(Dropout(dropout_rate))
        model.compile(loss = 'mse', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return model

    def _build_RNN(self, num_inputs, lookback, num_rnn_units, num_hidden_neurons, num_hidden_layers=1, num_output_neurons=4, dropout_rate=0.2):
        model = Sequential()
        model.add(SimpleRNN(num_rnn_units, input_length=lookback+1, input_dim=num_inputs, activation='tanh'))
        if num_hidden_layers == 0:
            model.add(Dense(units=self.num_actions, input_dim=num_inputs, activation='linear'))
            model.add(Dropout(dropout_rate))
        else:
            if type(num_hidden_neurons) is int:
                num_hidden_neurons = np.asarray(num_hidden_neurons * np.ones(num_hidden_layers), dtype=int)
            else:
                num_hidden_layers = len(num_hidden_neurons)
            model.add(Dense(units=num_hidden_neurons[0], input_dim=num_inputs, activation='relu'))
            model.add(Dropout(dropout_rate))
            for layer in range(0, num_hidden_layers - 1):
                model.add(Dense(units=num_hidden_neurons[layer], activation='relu'))
                model.add(Dropout(dropout_rate))
            model.add(Dense(units=num_output_neurons, activation='linear'))
            model.add(Dropout(dropout_rate))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.model = model
        return model

    def make_decision(self, state):
        p = np.ones(self.num_actions)
        p *= (self.epsilon / (self.num_actions-1))
        if self.network_type == 'Q':
            q_values = self.model.predict(state, batch_size=1)
        else:
            q_values = []
            for action in self.actions:
                tmp=0
                for sensor_reading in self.sensor_readings:
                    next_state_unormalized = self.state_estimator.predict(action, sensor_reading).flatten()
                    normalizer = sum(next_state_unormalized)
                    normalized = (next_state_unormalized/normalizer)
                    immediate_reward = np.dot(self.immediate_rewards, normalized)
                    if self.recurrent:
                        tmp2 = state.copy()
                        tmp2.pop()
                        tmp2.append(np.append(normalized, np.zeros(self.num_actions)))
                        future_value = self.model.predict(np.reshape(tmp2, (1, len(tmp2), len(tmp2[0])))).flatten()
                    else:
                        future_value = self.model.predict(np.reshape(normalized,(1, len(normalized)))).flatten()
                    tmp+=(immediate_reward+future_value)*normalizer
                q_values.append(tmp)
            q_values = np.array(q_values)
        p[np.argmax(q_values)] = 1 - self.epsilon
        return np.random.choice(range(0, self.num_actions), p=p)


    def update(self, state, action, reward, next_state,completed=False, log=True):
        s = state
        ns = next_state

        if self.network_type == 'Q':
            future_q = np.max(self.model.predict(ns, batch_size=1)[0])
            q = self.model.predict(s, batch_size=1)[0]
            target = reward + self.gamma*future_q
            q[action] = target
            q = np.reshape(q, (1,len(q)))
            self.model.fit(s, q, batch_size=1, epochs=1, verbose=0)
        else:
            future_v = self.model.predict(ns, batch_size=1)[0]
            target = reward + self.gamma*future_v
            self.model.fit(s, target, batch_size=1, epochs=1, verbose=0)

        self.current_return += reward
        if log:
            self.log.append([state, action, next_state, reward, completed])


def qnn_test():
    num_colours = 2
    num_states = 49
    model_num = 2
    figsize = (7.5, 6)
    initial_belief = np.asarray(np.ones(num_states),dtype=float)/num_states
    connected = False
    reward_vector = np.ones(num_states)*-1
    reward_vector[num_states-1] = 0
    fig = plt.figure(figsize=figsize)
    while not connected:
        map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
        map.show(delay=0.5)
        plt.close()
        connected = map.has_path(0, num_states-1)

    model = transition_model = map.get_transition_model(noise=0)
    #transition_model = get_transition_model(map, model)
    sensor_model = map.get_sensor_model(noise=0.2)

    pomdp = POMDP(num_states=num_states, num_actions=4, transition_model=model, sensor_model=sensor_model, initial_belief=initial_belief, num_hidden_neurons=[256, 512], num_hidden_layers=2 , epsilon=0.5,epsilon_decay=0.99995)

    prev_belief = initial_belief
    prev_state = 0
    iteration = 0
    print('Iteration {}'.format( iteration))
    moves = 0
    visualize = 0
    returns = []
    show =1
    save =1
    save_titles = []
    show_graph = False
    fig = plt.figure(figsize=figsize)

    while iteration<=50000:
        moves += 1
        cardinals = ['N','E','W','S']
        action = pomdp.make_decision(np.reshape(prev_belief, (1, num_states)))
        card_action = cardinals[action]
        actual_state = int(np.random.choice(range(0, map.num_states), p=transition_model(card_action)[prev_state]))
        sensor_reading = map.colour_map[actual_state]
        current_belief = pomdp.state_estimator.update(card_action, sensor_reading)

        if visualize:
            save_title = 'move_{}'.format(moves)
            map.show(actual_state=actual_state,node_weights= current_belief, delay=0.05, title='Iteration {}\n Current Score: {}\n Last Action Taken: {}\n Move: {}'.format(iteration, pomdp.current_return, card_action, moves), show=show, save=save, save_title=save_title, fig=fig, figsize=figsize)
            save_titles.append(save_title)

        if actual_state == map.num_states-1:
            reward = 10
            moves = 0
            pomdp.update(np.reshape(prev_belief,(1,num_states)), action, reward, current_belief)

            if visualize:
                actual_state = 0
            else:
                connected = False
                while not connected:
                    actual_state = np.random.choice(range(map.num_states))
                    connected = map.has_path(actual_state, num_states - 1)

            pomdp.state_estimator.posterior = initial_belief
            pomdp.epsilon *= pomdp.epsilon_decay
            returns.append(pomdp.current_return)
            pomdp.current_return = 0

            if show_graph:
                draw_graph(returns, iteration, fig)

            if visualize:
                create_gif(image_titles=save_titles, num_states=num_states, iteration=iteration)

            iteration += 1
            print('Iteration {}'.format(iteration))
            save_titles = []
            #
            # if iteration%100 ==0:
            #     connected=False
            #     while not connected:
            #         map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
            #         map.show(delay=0.5, show=0)
            #         connected = map.has_path(0, num_states - 1)
            #     model = map.get_transition_model(noise=0)
            #     transition_model = get_transition_model(map, model)
            #     sensor_model = map.get_sensor_model(noise=0.2)
            #     pomdp.state_estimator.transition_model = model
            #     pomdp.state_estimator.sensor_model = sensor_model


        elif(moves >= 50):
            moves = 0
            reward = -1
            pomdp.update(np.reshape(prev_belief, (1, num_states)), action, reward, current_belief)

            if visualize:
                actual_state = 0
            else:
                connected = False
                while not connected:
                    actual_state = np.random.choice(range(map.num_states))
                    connected = map.has_path(actual_state, num_states - 1)

            pomdp.state_estimator.posterior = initial_belief
            pomdp.epsilon *= pomdp.epsilon_decay
            returns.append(pomdp.current_return)
            pomdp.current_return = 0

            if show_graph:
                draw_graph(returns, iteration, fig)
            if visualize:
                create_gif(image_titles=save_titles, num_states=num_states, iteration=iteration)

            iteration += 1
            print('Iteration {}'.format(iteration))
            save_titles = []
            #
            # if iteration % 100 == 0:
            #     connected=False
            #     while not connected:
            #         map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
            #         map.show(delay=0.5, show=0)
            #         connected = map.has_path(0, num_states - 1)
            #     model = map.get_transition_model(noise=0)
            #     transition_model = get_transition_model(map, model)
            #     sensor_model = map.get_sensor_model(noise=0.2)
            #     pomdp.state_estimator.transition_model = model
            #     pomdp.state_estimator.sensor_model = sensor_model


        elif not map.is_connected(actual_state, prev_state, True):
            reward = -2
            pomdp.update(np.reshape(prev_belief, (1, num_states)), action, reward, current_belief)
        else:
            reward = reward_vector.dot(current_belief)
            pomdp.update(np.reshape(prev_belief, (1, num_states)), action, reward, current_belief)

        prev_belief = current_belief
        prev_state = actual_state
        show_graph = iteration in range(0, 50000, 500)
        tmp = []
        for i in range(2000, 50000, 2000):
            tmp += list(range(i, i+3))
        visualize = iteration in tmp

    pomdp.model.save('qnn_{}_states_model_{}.h5'.format(num_states, model_num))

def vnn_test():
    num_colours = 2
    num_states = 16
    model_num = 2
    figsize = (18.5, 6)
    gs = GridSpec(2,2)
    cardinals = ['N', 'E', 'W', 'S']
    initial_belief = np.asarray(np.ones(num_states),dtype=float)/num_states
    connected = False
    reward_vector = np.ones(num_states)*-1
    reward_vector[num_states-1] = 0
    fig = plt.figure(2,figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    while not connected:
        map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
        map.show(delay=1,ax=ax)
        plt.close()
        connected = map.has_path(0, num_states-1)

    model = transition_model = map.get_transition_model(noise=0)
    # transition_model = get_transition_model(map, model)
    sensor_model = map.get_sensor_model(noise=0.2)
    sensor_readings = np.unique(map.colour_map)
    sensor_probabilities = np.array([float(sum(reading == sensor_readings))/len(sensor_readings) for reading in sensor_readings])
    pomdp = POMDP(num_states=num_states, num_actions=4, transition_model=model, sensor_model=sensor_model, initial_belief=initial_belief, immediate_rewards=reward_vector, num_hidden_neurons=[64, 128], num_hidden_layers=2, num_output_neurons=1, epsilon=0.4,epsilon_decay=0.99995,network_type='V',sensor_readings=sensor_readings,sensor_probabilities=sensor_probabilities, actions =cardinals )
    prev_belief = initial_belief
    prev_state = 0
    iteration = 0
    print('Iteration {}'.format( iteration))
    moves = 0
    visualize = 0
    returns = []
    show =1
    save =1
    save_titles = []
    show_graph = False
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(gs[:, 1])
    while iteration<=50000:
        moves += 1
        action = pomdp.make_decision(np.reshape(prev_belief,(1, num_states)))
        card_action = cardinals[action]
        actual_state = int(np.random.choice(range(0, map.num_states), p=transition_model(card_action)[prev_state]))
        sensor_reading = map.colour_map[actual_state]
        current_belief = pomdp.state_estimator.update(card_action, sensor_reading)

        if visualize:
            save_title = 'move_{}'.format(moves)
            map.show(actual_state=actual_state,node_weights= current_belief, delay=0.05, title='Iteration {}\n Current Score: {}\n Last Action Taken: {}\n Move: {}'.format(iteration, pomdp.current_return, card_action, moves), show=show, save=save, save_title=save_title, fig=fig, figsize=figsize, ax=ax)
            save_titles.append(save_title)

        if actual_state == map.num_states-1:
            reward = 10
            moves=0
            pomdp.update(np.reshape(prev_belief,(1,num_states)), action, reward, current_belief)

            if visualize:
                actual_state = 0
            else:
                connected = False
                while not connected:
                    actual_state = np.random.choice(range(map.num_states))
                    connected = map.has_path(actual_state, num_states - 1)

            pomdp.state_estimator.posterior = initial_belief
            pomdp.epsilon *= pomdp.epsilon_decay
            returns.append(pomdp.current_return)
            pomdp.current_return = 0

            if show_graph:
                draw_graph(returns, iteration, fig, gs)

            if visualize:
                create_gif(image_titles=save_titles, num_states=num_states, iteration=iteration)

            iteration += 1
            print('Iteration {}'.format(iteration))
            save_titles = []
            #
            # if iteration%100 ==0:
            #     connected=False
            #     while not connected:
            #         map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
            #         map.show(delay=0.5, show=0)
            #         connected = map.has_path(0, num_states - 1)
            #     model = map.get_transition_model(noise=0)
            #     transition_model = get_transition_model(map, model)
            #     sensor_model = map.get_sensor_model(noise=0.2)
            #     pomdp.state_estimator.transition_model = model
            #     pomdp.state_estimator.sensor_model = sensor_model


        elif(moves >= 20):
            moves = 0
            reward = -1
            pomdp.update(np.reshape(prev_belief,(1,num_states)), action, reward, current_belief)

            if visualize:
                actual_state = 0
            else:
                connected = False
                while not connected:
                    actual_state = np.random.choice(range(map.num_states))
                    connected = map.has_path(actual_state, num_states - 1)

            pomdp.state_estimator.posterior = initial_belief
            pomdp.epsilon *= pomdp.epsilon_decay
            returns.append(pomdp.current_return)
            pomdp.current_return = 0

            if show_graph:
                draw_graph(returns, iteration, fig, gs)
            if visualize:
                create_gif(image_titles=save_titles, num_states=num_states, iteration=iteration)

            iteration += 1
            print('Iteration {}'.format(iteration))
            save_titles = []
            #
            # if iteration % 100 == 0:
            #     connected=False
            #     while not connected:
            #         map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
            #         map.show(delay=0.5, show=0)
            #         connected = map.has_path(0, num_states - 1)
            #     model = map.get_transition_model(noise=0)
            #     transition_model = get_transition_model(map, model)
            #     sensor_model = map.get_sensor_model(noise=0.2)
            #     pomdp.state_estimator.transition_model = model
            #     pomdp.state_estimator.sensor_model = sensor_model


        elif not map.is_connected(actual_state, prev_state, True):
            reward = -2
            pomdp.update(np.reshape(prev_belief, (1, num_states)), action, reward, current_belief)
        else:
            reward = reward_vector.dot(current_belief)
            pomdp.update(np.reshape(prev_belief, (1, num_states)), action, reward, current_belief)

        prev_belief = current_belief
        prev_state = actual_state
        show_graph = iteration in range(0, 50000, 500)
        tmp = []
        for i in range(2000, 50000, 2000):
            tmp += list(range(i, i+3))
        visualize = iteration in tmp

    pomdp.model.save('vnn_{}_states_model_{}.h5'.format(num_states, model_num))


def QRN_test():
    num_colours = 2
    num_states = 49
    lookback = 1
    model_num = 1
    figsize = (18.5, 6)
    gs = GridSpec(2,2)
    cardinals = ['N', 'E', 'W', 'S']
    initial_belief = np.asarray(np.ones(num_states),dtype=float)/num_states
    connected = False
    reward_vector = np.ones(num_states)*-1
    reward_vector[num_states-1] = 0
    fig = plt.figure(2, figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    while not connected:
        map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
        map.show(delay=1,ax=ax)
        plt.close()
        connected = map.has_path(0, num_states-1)

    model = transition_model = map.get_transition_model(noise=0)
    # transition_model = get_transition_model(map, model)
    sensor_model = map.get_sensor_model(noise=0.2)
    sensor_readings = np.unique(map.colour_map)
    sensor_probabilities = np.array([float(sum(reading == sensor_readings))/len(sensor_readings) for reading in sensor_readings])
    pomdp = POMDP(num_states=num_states, num_actions=4, transition_model=model, sensor_model=sensor_model, initial_belief=initial_belief, immediate_rewards=reward_vector, lookback=1,recurrent=True, num_rnn_units=64, num_hidden_neurons=[128], num_hidden_layers=1, num_output_neurons=4, epsilon=0.4,epsilon_decay=0.9995,network_type='Q',sensor_readings=sensor_readings,sensor_probabilities=sensor_probabilities, actions =cardinals )
    prev_belief = initial_belief
    prev_state = 0
    iteration = 0
    print('Iteration {}'.format( iteration))
    moves = 0
    visualize = 0
    returns = []
    show =1
    save =1
    save_titles = []
    show_graph = False
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(gs[:, 1])
    prev_beliefs = deque(np.zeros([lookback+1, num_states]), maxlen=lookback+1)
    prev_beliefs.append(initial_belief)
    logs = []
    while iteration<=20000:
        moves += 1

        action = pomdp.make_decision(np.reshape(list(prev_beliefs), (1, lookback+1, num_states)))
        card_action = cardinals[action]
        actual_state = int(np.random.choice(range(0, map.num_states), p=transition_model(card_action)[prev_state]))
        sensor_reading = map.colour_map[actual_state]
        current_belief = pomdp.state_estimator.update(card_action, sensor_reading)

        if visualize:
            save_title = 'move_{}'.format(moves)
            map.show(actual_state=actual_state,node_weights= current_belief, delay=0.05, title='Iteration {}\n Current Score: {}\n Last Action Taken: {}\n Move: {}'.format(iteration, pomdp.current_return, card_action, moves), show=show, save=save, save_title=save_title, fig=fig, figsize=figsize, ax=ax)
            save_titles.append(save_title)

        if actual_state == map.num_states-1:
            reward = 10
            moves=0
            current_beliefs = prev_beliefs.copy()
            current_beliefs.append(current_belief)
            pomdp.update(np.reshape(list(prev_beliefs), (1, lookback+1, num_states)), action, reward, np.reshape(list(current_beliefs), (1, lookback+1, num_states)))

            if visualize:
                actual_state = 0
            else:
                connected = False
                while not connected:
                    actual_state = np.random.choice(range(map.num_states))
                    connected = map.has_path(actual_state, num_states - 1)

            pomdp.state_estimator.posterior = initial_belief
            pomdp.epsilon *= pomdp.epsilon_decay
            returns.append(pomdp.current_return)
            pomdp.current_return = 0

            if show_graph:
                draw_graph(returns, iteration, fig, gs)

            if visualize:
                create_gif(image_titles=save_titles, num_states=num_states, iteration=iteration)

            iteration += 1
            print('Iteration {}'.format(iteration))
            save_titles = []
            #
            # if iteration%100 ==0:
            #     connected=False
            #     while not connected:
            #         map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
            #         map.show(delay=0.5, show=0)
            #         connected = map.has_path(0, num_states - 1)
            #     model = map.get_transition_model(noise=0)
            #     transition_model = get_transition_model(map, model)
            #     sensor_model = map.get_sensor_model(noise=0.2)
            #     pomdp.state_estimator.transition_model = model
            #     pomdp.state_estimator.sensor_model = sensor_model


        elif(moves >= 50):
            moves = 0
            reward = -1
            current_beliefs = prev_beliefs.copy()
            current_beliefs.append(current_belief)
            pomdp.update(np.reshape(prev_beliefs, (1, lookback+1, num_states)), action, reward, np.reshape(current_beliefs, (1, lookback+1, num_states)))

            if visualize:
                actual_state = 0
            else:
                connected = False
                while not connected:
                    actual_state = np.random.choice(range(map.num_states))
                    connected = map.has_path(actual_state, num_states - 1)

            pomdp.state_estimator.posterior = initial_belief
            pomdp.epsilon *= pomdp.epsilon_decay
            returns.append(pomdp.current_return)
            pomdp.current_return = 0

            if show_graph:
                draw_graph(returns, iteration, fig, gs)
            if visualize:
                create_gif(image_titles=save_titles, num_states=num_states, iteration=iteration)

            iteration += 1
            print('Iteration {}'.format(iteration))
            save_titles = []
            #
            # if iteration % 100 == 0:
            #     connected=False
            #     while not connected:
            #         map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
            #         map.show(delay=0.5, show=0)
            #         connected = map.has_path(0, num_states - 1)
            #     model = map.get_transition_model(noise=0)
            #     transition_model = get_transition_model(map, model)
            #     sensor_model = map.get_sensor_model(noise=0.2)
            #     pomdp.state_estimator.transition_model = model
            #     pomdp.state_estimator.sensor_model = sensor_model

        elif not map.is_connected(actual_state, prev_state, True):
            reward = -2
            current_beliefs = prev_beliefs.copy()
            current_beliefs.append(current_belief)
            pomdp.update(np.reshape(prev_beliefs, (1, lookback+1, num_states)), action, reward, np.reshape(current_beliefs, (1, lookback+1, num_states)))
        else:
            reward = reward_vector.dot(current_belief)
            current_beliefs = prev_beliefs.copy()
            current_beliefs.append(current_belief)
            pomdp.update(np.reshape(prev_beliefs, (1, lookback+1, num_states)), action, reward, np.reshape(current_beliefs, (1, lookback+1, num_states)))

        prev_beliefs = current_beliefs
        prev_state = actual_state
        show_graph = iteration in range(0, 50000, 500)
        tmp = []
        for i in range(2000, 50000, 2000):
            tmp += list(range(i, i+3))
        visualize = iteration in tmp

    pomdp.model.save('qrn_{}_states_model_{}.h5'.format(num_states, model_num))
    pomdp.log.to_csv('qrn_model.csv')

def VRN_test():
    num_colours = 2
    num_states = 16
    lookback = 4
    model_num = 1
    figsize = (18.5, 6)
    gs = GridSpec(2,2)
    cardinals = ['N', 'E', 'W', 'S']
    initial_belief = np.asarray(np.ones(num_states),dtype=float)/num_states
    connected = False
    reward_vector = np.ones(num_states)*-1
    reward_vector[num_states-1] = 0
    fig = plt.figure(2, figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    while not connected:
        map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
        map.show(delay=1,ax=ax)
        plt.close()
        connected = map.has_path(0, num_states-1)

    model = transition_model = map.get_transition_model(noise=0)
    # transition_model = get_transition_model(map, model)
    sensor_model = map.get_sensor_model(noise=0.2)
    sensor_readings = np.unique(map.colour_map)
    sensor_probabilities = np.array([float(sum(reading == sensor_readings))/len(sensor_readings) for reading in sensor_readings])
    pomdp = POMDP(num_states=num_states, num_actions=4, transition_model=model, sensor_model=sensor_model, initial_belief=initial_belief, immediate_rewards=reward_vector, lookback=lookback,recurrent=True,num_inputs=num_states+4, num_rnn_units=128, num_hidden_neurons=[256, 512], num_hidden_layers=2, num_output_neurons=1, epsilon=0.4,epsilon_decay=0.99995,network_type='V',sensor_readings=sensor_readings,sensor_probabilities=sensor_probabilities, actions =cardinals )
    prev_belief = initial_belief
    prev_state = 0
    iteration = 0
    print('Iteration {}'.format( iteration))
    moves = 0
    visualize = 0
    returns = []
    show =1
    save =1
    save_titles = []
    show_graph = False
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(gs[:, 1])
    prev_beliefs = deque(np.zeros([lookback+1, num_states+4]), maxlen=lookback+1)
    prev_beliefs.append(np.append(initial_belief, np.zeros(4)))
    success = []
    while iteration <= 40000:
        moves += 1

        action = pomdp.make_decision(list(prev_beliefs))
        card_action = cardinals[action]
        actual_state = int(np.random.choice(range(0, map.num_states), p=transition_model(card_action)[prev_state]))
        sensor_reading = map.colour_map[actual_state]
        current_belief = pomdp.state_estimator.update(card_action, sensor_reading)
        encoded_action = action_encoder(card_action)

        if visualize:
            save_title = 'move_{}'.format(moves)
            map.show(actual_state=actual_state,node_weights= current_belief, delay=0.05, title='Iteration {}\n Current Score: {}\n Last Action Taken: {}\n Move: {}'.format(iteration, pomdp.current_return, card_action, moves), show=show, save=save, save_title=save_title, fig=fig, figsize=figsize, ax=ax)
            save_titles.append(save_title)
        pure_belief = current_belief
        current_belief = np.append(current_belief, encoded_action)

        if actual_state == map.num_states-1:
            reward = 10
            moves=0
            current_beliefs = prev_beliefs.copy()
            current_beliefs.append(current_belief)
            pomdp.update(np.reshape(list(prev_beliefs), (1, lookback+1, num_states+4)), action, reward, np.reshape(list(current_beliefs), (1, lookback+1, num_states+4)))

            if visualize:
                actual_state = 0
            else:
                connected = False
                while not connected:
                    actual_state = np.random.choice(range(map.num_states))
                    connected = map.has_path(actual_state, num_states - 1)

            pomdp.state_estimator.posterior = initial_belief
            pomdp.epsilon *= pomdp.epsilon_decay
            success.append(1)
            returns.append(pomdp.current_return)
            pomdp.current_return = 0

            if show_graph:
                draw_graph(returns,success, iteration, fig, gs)

            if visualize:
                create_gif(image_titles=save_titles, num_states=num_states, iteration=iteration)

            iteration += 1
            print('Iteration {}'.format(iteration))
            save_titles = []

            if iteration%100 ==0:
                connected=False
                while not connected:
                    map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
                    map.show(delay=0.5, show=1, ax=ax)
                    connected = map.has_path(0, num_states - 1)
                transition_model = map.get_transition_model(noise=0)
                sensor_model = map.get_sensor_model(noise=0.2)
                pomdp.state_estimator.transition_model = transition_model
                pomdp.state_estimator.sensor_model = sensor_model


        elif(moves >= 30):
            moves = 0
            reward = -1
            current_beliefs = prev_beliefs.copy()
            current_beliefs.append(current_belief)
            pomdp.update(np.reshape(prev_beliefs, (1, lookback+1, num_states+4)), action, reward, np.reshape(current_beliefs, (1, lookback+1, num_states+4)))

            if visualize:
                actual_state = 0
            else:
                connected = False
                while not connected:
                    actual_state = np.random.choice(range(map.num_states))
                    connected = map.has_path(actual_state, num_states - 1)

            pomdp.state_estimator.posterior = initial_belief
            pomdp.epsilon *= pomdp.epsilon_decay
            returns.append(pomdp.current_return)
            success.append(0)
            pomdp.current_return = 0

            if show_graph:
                draw_graph(returns, success, iteration, fig, gs)
            if visualize:
                create_gif(image_titles=save_titles, num_states=num_states, iteration=iteration)

            iteration += 1
            print('Iteration {}'.format(iteration))
            save_titles = []

            if iteration % 100 == 0:
                connected=False
                while not connected:
                    map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
                    map.show(delay=0.5, show=1, ax=ax)
                    connected = map.has_path(0, num_states - 1)
                transition_model = map.get_transition_model(noise=0)
                sensor_model = map.get_sensor_model(noise=0.2)
                pomdp.state_estimator.transition_model = transition_model
                pomdp.state_estimator.sensor_model = sensor_model

        elif not map.is_connected(actual_state, prev_state, True):
            reward = -2
            current_beliefs = prev_beliefs.copy()
            current_beliefs.append(current_belief)
            pomdp.update(np.reshape(prev_beliefs, (1, lookback+1, num_states+4)), action, reward, np.reshape(current_beliefs, (1, lookback+1, num_states+4)))
        else:
            reward = reward_vector.dot(pure_belief)
            current_beliefs = prev_beliefs.copy()
            current_beliefs.append(current_belief)
            pomdp.update(np.reshape(prev_beliefs, (1, lookback+1, num_states+4)), action, reward, np.reshape(current_beliefs, (1, lookback+1, num_states+4)))

        prev_beliefs = current_beliefs
        prev_state = actual_state
        show_graph = iteration in range(0, 50000, 100)
        tmp = []
        for i in range(200, 50000, 200):
            tmp += list(range(i, i+3))
        visualize = iteration in tmp

    pomdp.model.save('qrn_{}_states_model_{}.h5'.format(num_states, model_num))
    pomdp.log.to_csv('qrn_model.csv')


def action_encoder(action):
    return action == np.array(['N', 'E', 'W', 'S'])


def draw_graph(returns, success, iteration, fig, gs):
    returns = np.array(returns)
    success = np.array(success)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    ave_returns = [np.mean((returns[i - 50:i])) for i in range(50, len(returns), 50)]
    ind = list(range(50, len(returns), 50))
    ax1.cla()
    ax1.plot(ind, np.array(ave_returns))
    ax1.set_ylabel('Returns')
    ax1.set_xlabel('Iteration')

    percentage = [sum(1==(success[(i - 100) : i]))/100 for i in range(100, len(success), 100)]
    if len(percentage) != 0:
        ind = list(range(100, len(success), 100))
        ax2.cla()
        ax2.plot(ind, percentage)
        ax2.set_title('Percentage wins over past 100 iterations')
        ax2.set_ylabel('Percentage')
        ax2.set_xlabel('Iteration')

    plt.savefig('Media/Graphs/iteration_{}_graph'.format(iteration))
    plt.show()
    fig.set_size_inches(18.5, 6)
    plt.pause(0.1)


def create_gif(image_titles, num_states, iteration):
    images = []
    for image_title in image_titles:
        im = imageio.imread('Media/'+image_title+'.png')
        images.append(im)
    imageio.mimwrite(os.path.dirname(__file__) + r'/Media/GIFs/{}_states_model_iteration_{}.gif'.format(num_states, iteration), np.array(images), duration =0.8)

def get_transition_model(map, model):
    tmp = dict()
    for action in ['N', 'S', 'E', 'W']:
        for state in range(0, map.num_states):
            if state in tmp:
                d = tmp[state]
            else:
                d = dict()
            d[action] = model(action)[state]
            tmp[state] = d
    actions = []

    for _ in range(0, map.num_states):
        actions.append(['S', 'E', 'W', 'N'])
    return lambda a, st: tmp[a].values() if st else model(a)


if __name__ == '__main__':
    VRN_test()
