import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('..//..//KaSeDy//pybot'))
from tools.Map import Map

from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.regularizers import l2

from pandas import DataFrame, concat
from collections import deque
from itertools import product
from time import time

import matplotlib.pyplot as plt
import imageio
from matplotlib.gridspec import GridSpec


#recurrent_layer=0,num_inputs=None, num_hidden_neurons=20, num_hidden_layers=1, lookback=1, num_rnn_units=64, num_output_neurons=4, convolutional_layer=0, filters=16, kernel_size=2, dropout_rate=0.2, l2=0.7

class MDP:
    """Basic implementation of a finite Markov Decision Process    
    """

    def __init__(self, num_states=None, num_actions=None, gamma=0.9,lr=1e-3, method ='q-network', policy_type='epsilon-greedy', epsilon = 1e-2, epsilon_decay=0.995, minimum_epsilon=1e-2,
                 transition_model=None, immediate_rewards=None, value_function=None, function_type=0, policy=None, init_Q_matrix=None, q_model=None, v_model=None, network_type='Q', folder_name='Default',log=None, state_formatter=None, **kwargs ):
        """
        Initializer for MDP agent

        Parameters
        ----------

        num_states:        int - state space size
        num_actions:       int - action space size
        gamma:             float - discount parameter used in updates
        lr:                float - learning rate
        method:            string - method used to determine policy (Case insensitive). Options ['vi', 'pi', 'q-table', 'q-network', 'policy-network']
        epsilon:           float - epsilon used in epsilon-greedy policy
        epsilon-decay:     float - epsilon is discounted by this amount after each episode
        minimum_epsilon:   smallest value epsilon is allowed to take
        transition_model:  f: int -> int [num_states, num_states] - function that returns float [num_states, num_states] (transition probability from each state to another state) when given action
        immediate_rewards: f: int x int -> float - function that returns float (reward) when given state index and action index
        value_funtion:     f: int -> float - function that returns value of state when given state index
        function_type:     int - Index indicating function type
        policy:            f: int -> float [num_actions, 1] - function that returns probabilities of choosing an action when given a state index
        init_Q_matrix:     float[num_states, num_actions] - initial Q matrix
        q-model:           keras neural-network -  approximates the Q matrix
        v-model:           keras neural-network -  approximates the V function
        network_type:      String - Type of neural-network (Case insensitive). Options ['Q', 'V']
        log:               DataFrame - Dataframe used to store episode info
        folder:            String- name of the folder to store data saved from agent
        state_formatter:   Function that formats states
        **kwargs:          Arguments passed to _build_NN on how to build the neural-network
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.transition_model = transition_model
        self.immediate_rewards = immediate_rewards
        self.value_function = value_function
        self.function_type = function_type
        self.policy = policy
        self.policy_type = policy_type.lower()
        self.Q_matrix = init_Q_matrix
        self.epsilon = epsilon
        self.minimum_epsilon = minimum_epsilon
        self.gamma = gamma
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        if q_model is None and v_model is None:
                self.model = self._build_NN(**kwargs) #Q-Neural Network
        elif q_model is not None:
            self.model = q_model
        else:
            self.model = v_model
        self.network_type = network_type.capitalize()
        self .method = method.lower()
        self.current_return = 0
        self.buffer = DataFrame(columns=['Previous State', 'Action Taken', 'Current State', 'Reward', 'Completed'])
        if log is None:
            self.log = DataFrame(columns=['Previous State', 'Action Taken', 'Current State', 'Reward', 'Completed', 'Total Return'])
        else:
            self.log = log
        self.foldername = folder_name
        f=lambda x:x
        self.state_formatter = f if state_formatter is None else state_formatter

    def train_offline(self, transition_model=None, immediate_returns=None, method='vi', gamma=0.9, theta=1e-10, alpha=0.8, terminal_states=[], in_place=True, initial_policy=None):
        method = method.strip().lower()
        if method in ['vi', 'pi' ] and (transition_model is None or immediate_returns is None):
            raise Exception(
                'Transition probabilities and immediate returns cannot be none when using value iteration or policy iteration.'
            )
        if method == 'vi':
            self.function_type = 0
            self.value_function, self.policy = self.value_iteration(transition_model, immediate_returns, gamma, theta, terminal_states, in_place)

        if method == 'pi':
            self.function_type = 0
            self.value_function, self.policy = self.policy_iteration(initial_policy, transition_model, immediate_returns, gamma, theta, terminal_states, in_place)

        if method == 'q_table':
            self.num_states, self.num_actions = np.size(self.Q_matrix, 0), np.size(self.Q_matrix, 1)
            self.function_type = 1
            self.alpha = alpha
            self.gamma = gamma

    def q_matrix_update(self, state, action, reward, next_state):
        self.Q_matrix[state][action] += self.lr * (reward + self.gamma * max(self.Q_matrix[next_state]) - self.Q_matrix[state][action])

    def _build_NN(self, num_inputs=None, num_hidden_neurons=10, num_hidden_layers=1, num_output_neurons=4, dropout_rate=0.2, lr=0.001, recurrent_layer=0, num_rnn_units=128, lookback=1, convolutional_layer=0, filters=16, kernel_size=2, l2_reg=0.7):
        num_inputs = self.num_states if num_inputs is None else num_inputs
        model = Sequential()
        num_inputs = self.num_states if num_inputs is None else num_inputs
        if not recurrent_layer:
            lookback=0
        model.add(InputLayer(input_shape=(lookback+1, num_inputs)))
        if convolutional_layer:
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
            model.add(MaxPooling1D(2))
        while recurrent_layer>1:
            model.add(LSTM(num_rnn_units, activation='relu', recurrent_dropout=dropout_rate,))
            recurrent_layer -= 1
        if recurrent_layer == 1:
            model.add(LSTM(num_rnn_units, activation='relu', recurrent_dropout=dropout_rate))
        if num_hidden_layers == 0:
            model.add(Dense(units=self.num_actions, input_dim=num_inputs, activation='linear', kernel_regularizer=l2(l2_reg)))
            model.add(Dropout(dropout_rate))
        else:
            if type(num_hidden_neurons) is int:
                num_hidden_neurons = np.asarray(num_hidden_neurons*np.ones(num_hidden_layers), dtype=int)
            else:
                num_hidden_layers = len(num_hidden_neurons)
            if not(convolutional_layer or recurrent_layer):
                model.add(Dense(units=num_hidden_neurons[0], input_dim=num_inputs, activation='relu', kernel_regularizer=l2(l2_reg)))
                model.add(Dropout(dropout_rate))
                num_hidden_layers-=1
            for layer in range(0, num_hidden_layers):
                model.add(Dense(units=num_hidden_neurons[layer], activation='relu', kernel_regularizer=l2(l2_reg)))
                model.add(Dropout(dropout_rate))
            model.add(Dense(units=num_output_neurons, activation='linear', kernel_regularizer=l2(l2_reg)))
        model.compile(loss='mse', optimizer=Adam(lr=lr), metrics=['accuracy'], )
        self.model = model
        return model

    def get_state_index(self, full_state):
        position, orientation, readings = full_state[0:self.num_states], full_state[self.num_states:self.num_states+4], full_state[self.num_states+4::]
        dic = dict(enumerate(product(range(len(position)), range(4), range(8))))
        dic = dict(zip(dic.values(), dic.keys()))
        convert_readings = lambda x: x[0]*(2**2) + x[1]*2 + x[0]
        position = np.argmax(position)
        orientation = np.argmax(orientation)
        readings = convert_readings(readings)
        return dic[(position, orientation, int(readings))]

    def make_decision(self, state):
        state = self.state_formatter(state)
        if self.policy_type =='random':
            return np.random.choice(range(self.num_actions))
        elif self.policy_type == 'epsilon-greedy':
            if np.random.rand() <= self.epsilon:
                return np.random.choice(range(self.num_actions))
            if self.method in ['q-network','policy-network']:
                q_values = np.array(self.model.predict(state)).flatten()
            else:
                q_values = self.Q_matrix[self.get_state_index(state)]
            return np.argmax(q_values)

        elif self.policy_type == 'softmax':
            if self.method == 'policy-network':          
                p = np.array(self.model.predict(state)).flatten()

            elif self.method == 'q-network':        
                tmp = np.exp(np.array(self.model.predict(state).flatten()))              
                p = tmp / np.sum(tmp)
                
            elif self.method == 'q-table':
                tmp = np.exp(self.Q_matrix[self.get_state_index(state)])
                p = tmp/np.sum(tmp)
                
            return np.random.choice(range(len(p)), p=p)


    def update(self, state, action, reward, next_state, done=False, log=True, replay=False, batch_size=1,minibatch_size=1, dropout=0.2):  
        self.current_return += reward
        state = self.state_formatter(state)
        next_state = self.state_formatter(next_state)
        if log:
            self.buffer.loc[len(self.buffer), :] = [state, action, next_state, reward, done]
            if done:
                self.buffer['Total Return'] = self.buffer['Reward'].sum()
                self.log = concat([self.log, self.buffer], axis=0)
                self.buffer = DataFrame(columns=['Previous State', 'Action Taken', 'Current State', 'Reward', 'Completed'])
        if done:
            self.epsilon = np.max(self.epsilon**self.epsilon_decay, self.minimum_epsilon)

        if np.random.random_sample() < dropout:
            return

        if self.method == 'q_table':
            self.q_matrix_update(self.get_state_index(state), action, reward, self.get_state_index(next_state))

        elif self.method == 'q-network':
            
            if np.random.random_sample() < replay:
                targets = []
                states = []

                for _, frame in self.log.sample(len(self.log) if len(self.log) < batch_size else batch_size).iterrows():
                    state, action, next_state, reward, done,_ = frame
                    state = self.state_formatter(state)
                    next_state = self.state_formatter(next_state)

                    q = self.model.predict(state, batch_size=1).flatten()
                    if done:
                        target = reward
                    else:
                        future_q = np.max(self.model.predict(next_state, batch_size=1).flatten())
                        target = reward + self.gamma * future_q
                    q[action] = target
                    q = np.reshape(q, (-1, len(q)))
                    targets.append(q)
                    states.append(state)
                self.model.fit(np.array(states)[0], np.array(targets)[0], batch_size=minibatch_size, epochs=1, verbose=0)

            else:

                future_q = np.max(self.model.predict(next_state, batch_size=1).flatten())
                q = self.model.predict(state, batch_size=1).flatten()
                target = reward + self.gamma*future_q
                q[action] = target
                q = np.reshape(q, (-1, len(q)))
                self.model.fit(state, q, batch_size=1, epochs=1, verbose=0)

        elif self.method == 'policy-network':
                if np.random.random_sample() < replay:
                    targets = []
                    states = []

                    for _, frame in self.log.sample(len(self.log) if len(self.log) < batch_size else batch_size).iterrows():
                        state, action, next_state, reward, done = frame
                        q = self.model.predict(state, batch_size=1).flatten()
                        if done:
                            target = reward
                        else:
                            future_q = np.max(self.model.predict(next_state, batch_size=1).flatten())
                            target = reward + self.gamma * future_q
                        q[action] = target
                        q = np.reshape(q, (-1, len(q)))
                        targets.append(q)
                        states.append(state)
                    self.model.fit(np.array(states)[0], np.array(targets)[0], batch_size=minibatch_size, epochs=1, verbose=0)

                else:

                    future_q = np.max(self.model.predict(next_state, batch_size=1).flatten())
                    q = self.model.predict(state, batch_size=1).flatten()
                    target = reward + self.gamma*future_q
                    q[action] = target
                    q = np.reshape(q, (-1, len(q)))
                    self.model.fit(state, q, batch_size=1, epochs=1, verbose=0)

    def draw_graph(self, returns, success=None, iteration=None, fig=None, gs=None, ax=None):
        if success is not None:
            returns = np.array(returns)
            success = np.array(success)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])

            ave_returns = [np.mean((returns[i - 50:i])) for i in range(50, len(returns), 1)]
            ind = list(range(50, len(returns), 1))
            ax1.cla()
            ax1.plot(ind, np.array(ave_returns))
            ax1.set_ylabel('Returns')
            ax1.set_xlabel('Iteration')

            percentage = [sum(1 == (success[(i - 100): i])) / 100 for i in range(50, len(success), 1)]
            if len(percentage) != 0:
                ind = list(range(50, len(success), 1))
                ax2.cla()
                ax2.plot(ind, percentage)
                ax2.set_title('Percentage wins over past 100 iterations')
                ax2.set_ylabel('Percentage')
                ax2.set_xlabel('Iteration')

            plt.savefig('Media/Graphs/{}/iteration_{}_graph'.format(self.foldername, iteration))
            plt.show()
            fig.set_size_inches(18.5, 8)
            plt.pause(0.1)
        else:
            ax.cla()
            returns = np.array(returns)
            ave_returns = [np.mean((returns[i - 10:i])) for i in range(10, len(returns), 10)]
            ind = list(range(10, len(returns), 10))
            ax.plot(ind, ave_returns)
            ax.set_ylabel('Returns')
            ax.set_xlabel('Iteration')
            plt.pause(0.1)
            plt.show()

    def evaluate_model_in_environment(self, env, num_episodes, num_timesteps, show_env=0, show_graph=0, ax_env=None, ax_graph=None,
                                      verbose=1):
        """
        :param mdp: MDP agent who's model is being evaluated
        :param env: Environment the model is being evaluated on
        :param num_episodes: (int) Number of episodes to run for
        :param num_timesteps: (int) Nunmber of steps in each episode
        :param show_env(Boolean or list of integers): Flag indicating when and whether to render environment
        :param show_graph(Boolean or list of integers): Flag indicatin whether or when to plot graph of average returns
        :param ax_env axis to display environment on
        :param ax_graph axis to display graph on
        :param verbose: (int) Flag indicating what to print
        :return:
        """

        episodes = range(num_episodes)
        timesteps = range(num_timesteps)
        returns_list = deque(maxlen=100)
        average_returns = []

        if type(show_env) is int:
            visualize = [[], list(episodes)][show_env]
        else:
            visualize = list(show_env)
        episode_list = visualize.copy()
        visualize_env = lambda x: x in episode_list

        if type(show_graph) is int:
            visualize = [[], list(episodes)][show_graph]
        else:
            visualize = list(show_graph)
        episode_list_graph = visualize.copy()
        visualize_graph = lambda x: x in episode_list_graph

        current_return = 0
        returns = []
        for episode in episodes:
            prev_observation = env.reset()

            if visualize_env(episode):
                env.render()
            for t in timesteps:
                action = self.make_decision(prev_observation)
                observation, reward, done, _ = env.step(action)
                current_return += reward
                prev_observation = observation.copy()

                if visualize_env(episode):
                    env.render()
                if visualize_graph(episode):
                    self.draw_graph(returns, ax=ax_graph)
                if done:
                    returns_list.append(current_return)
                    returns.append(current_return)
                    if verbose == 1:
                        print('Episode {} finished after {} timesteps. Total reward {}. Average return over past 100 episodes {}'.format(episode, t, current_return, np.mean(list(returns_list))))
                    current_return = 0
                    break
        print('Average return: {.3f} +/- {.5f}'.format(np.mean(returns), np.std(returns)))

    @staticmethod
    def evaluate_maze_model(model=None, method='q-network', policy_type='softmax', train=0, complex_input=0, state_formatter = lambda x:x):
        num_colours = 2
        num_states = 9
        state_dim = num_states + 4
        num_actions = 3
        sensor_dim = 8

        end_state = num_states-1
        actions = ['LEFT_TURN', 'FORWARD', 'RIGHT_TURN']

        reward_vector = np.ones(num_states)*-1
        reward_vector[num_states-1] = 0
        epsilon = 0.1

        show = 1
        save = 1
        randomize = 1
        figsize = (18.5, 8)
        gs = GridSpec(2, 2)

        fig = plt.figure(1, figsize=figsize)
        ax = fig.add_subplot(gs[:, 1])
        map = generate_random_map(num_states, num_colours, 0, num_states-1, ax, delay=1 )

        transition_model = map.get_transition_model(noise=0)

        mdp = MDP(num_states=num_states, num_actions=num_actions, transition_model=transition_model, immediate_rewards=reward_vector,
                  q_model=model, policy_type=policy_type, network_type='Q', method=method, epsilon=epsilon, minimum_epsilon=epsilon, state_formatter=state_formatter)

        prev_state = np.zeros(state_dim)
        prev_state[num_states+1] = 1
        prev_state[0] = 1
        current_state = prev_state.copy()
        returns = []
        save_titles = []
        readings = np.reshape(readings_encoder(map.get_sensor_readings(np.argmax(prev_state), orientation=np.argmax([0, 1, 0, 0]))), [-1, sensor_dim])
        prev_full_state = np.append(current_state, readings)
        success = []

        for iteration in range(40000):
            print('Iteration {}'.format(iteration))
            show_graph = iteration in range(50, 50000, 50)
            tmp = []
            for i in range(0, 50000, 100):
                tmp += list(range(i, i + 3))
            visualize = iteration in tmp
            current_return = 0
            for moves in range(51):
                action_index = mdp.make_decision(prev_full_state)
                if policy_type =='epsilon-greedy':
                    if action_index == np.argmax(mdp.model.predict(mdp.state_formatter(prev_full_state))[0]):
                        random_or_greedy= 'greedy'
                    else:
                        random_or_greedy = 'random'
                else:
                    random_or_greedy=mdp.model.predict(mdp.state_formatter(prev_full_state))[0]
                    print(prev_full_state)
                    print(random_or_greedy)
                action = actions[action_index]
                prev_position = prev_state[0:num_states]
                if action == 'FORWARD':
                    card_action = action_to_cardinal(prev_state, num_states)
                    position = int(np.random.choice(range(0, map.num_states), p=transition_model(card_action)[np.argmax(prev_position)]))
                    current_state[num_states::] = prev_state[num_states::]
                    current_state[0:num_states] = position_encoder(position, num_states)
                else:
                    orientation = get_orientation(action, prev_state, num_states)
                    current_state[0:num_states] = prev_state[0:num_states]
                    current_state[num_states::] = orientation
                position = current_state[0:num_states]
                orientation = current_state[num_states::]
                readings = readings_encoder(map.get_sensor_readings(np.argmax(position), orientation=np.argmax(orientation)))

                if visualize:
                    save_title = 'move_{}'.format(moves)
                    orientation = current_state[num_states::]
                    map.show(actual_state=np.argmax(current_state), orientation=orientation, delay=0.05, title='Iteration {}\n Current Score: {}\n Last Action Taken: {}\n Move({}): {}\n Readings: {}'.format(iteration, current_return, action, random_or_greedy, moves, readings), show=show, save=save, save_title=save_title, fig=fig, figsize=figsize, ax=ax)
                    save_titles.append(save_title)
                    full_state = np.append(current_state, readings)

                if np.argmax(current_state) == end_state:
                    reward = 1
                    current_return += reward
                    current_full_state = full_state
                    if train:
                        mdp.update(prev_full_state, action_index, reward,current_full_state,
                                              current_full_state, done=1, replay=1,
                                   batch_size=moves, minibatch_size=5, dropout=0)

                    if visualize:
                        current_state = np.zeros(state_dim)
                        current_state[num_states + 1] = 1
                        current_state[0] = 1
                    else:
                        connected = False
                        while not connected:
                            current_state = np.zeros(state_dim)
                            current_state[num_states + np.random.choice(range(4))] = 1
                            index = np.random.choice(range(map.num_states))
                            current_state[index] = 1
                            connected = map.has_path(index, num_states - 1)


                    tmp = np.append(current_state, readings_encoder(map.get_sensor_readings(np.argmax(current_state), orientation=np.argmax([0, 1, 0, 0]))))
                    prev_full_state = tmp
                    prev_state = current_state
                    success.append(1)
                    break

                elif moves == 50:
                    reward = -0.1
                    current_return += reward
                    current_full_state = full_state

                    if train:
                        mdp.update(prev_full_state,
                                   action_index, reward,
                                   current_full_state, done=1, replay=1,
                                   batch_size=moves, minibatch_size=5, dropout=0)

                    connected = False
                    current_state = np.zeros(state_dim)
                    current_state[num_states + np.random.choice(range(4))] = 1

                    while not connected:
                        index = np.random.choice(range(map.num_states))
                        current_state[index] = 1
                        connected = map.has_path(index, num_states - 1)

                    prev_full_state = np.append(current_state, readings_encoder(map.get_sensor_readings(np.argmax(current_state), orientation=np.argmax([0, 1, 0, 0]))))
                    prev_state = current_state

                    success.append(0)
                    break

                elif np.argmax(prev_position) == np.argmax(position) and (action == 'FORWARD'):
                    reward = -0.2

                elif action == 'FORWARD':
                    pos_ind = np.argmax(position)
                    length = np.sqrt(num_states)
                    reward = -float(length - (pos_ind % length) + length - (pos_ind / length)) / (20 * length)
                else:
                    reward = -0.1

                current_return += reward
                current_full_state = full_state
                if train:
                    mdp.update(prev_full_state, action_index, reward, current_full_state, replay=0, dropout=1)

                prev_full_state = current_full_state
                prev_state = current_state

            returns.append(current_return)
            mdp.current_return = 0
            save_titles = []


            if show_graph:
                mdp.draw_graph(returns, success, iteration, fig, gs)

            if visualize:
                mdp.create_gif(image_titles=save_titles, num_states=num_states, iteration=iteration)

            if randomize and iteration % 10 == 0:
                map = generate_random_map(num_states, num_colours, np.argmax(current_state[0:num_states]), num_states - 1, ax, delay=0.5)
                transition_model = map.get_transition_model(noise=0)

    def create_gif(self, image_titles, num_states, iteration):
        images = []
        for image_title in image_titles:
            im = imageio.imread('Media/'+image_title+'.png')
            images.append(im)
        imageio.mimwrite(os.path.dirname(__file__) + r'/Media/GIFs/{}/{}_states_model_iteration_{}.gif'.format(self.foldername, num_states, iteration), np.array(images), duration =0.8)


    @staticmethod
    def value_iteration(transition_probabilities, immediate_rewards, actions, gamma=0.9, theta=0.1,
                        terminal_states=[], in_place=True, initial_policy=None):
        num_states = np.size(immediate_rewards, 0)
        if type(actions) is int:
            num = actions
            actions = []
            for _ in range(num):
                actions.append(range(0, num))
        values = np.random.rand(num_states)
        values[-1] = immediate_rewards[-1]
        if in_place:
            new_values = np.array([num_states, 1])
        if initial_policy is None:
            policy = []
            for state_actions in actions:
                policy.append(np.eye(1, len(state_actions)))
        else:
            policy = initial_policy
        while True:
            delta = -np.inf
            for i in range(0, num_states):
                if i in terminal_states:
                    continue
                tmp = np.dot(transition_probabilities(i, True), (immediate_rewards + gamma * values))
                new_v = np.max(tmp)
                policy[i] = np.zeros(len(actions[i]))
                policy[i][np.argmax(tmp)] = 1
                delta = max(delta, values[i] - new_v)
                if not in_place:
                    new_values[i] = new_v
                else:
                    values[i] = new_v
            if not in_place:
                values = new_values
            if delta < theta:
                break
        return values, policy
    @staticmethod
    def policy_evaluation(policy, transition_model, immediate_rewards, actions, gamma=0.9, theta=0.1, terminal_states=[], in_place=True):
        num_states = np.size(immediate_rewards, 0)
        if type(actions) is int:
            num = actions
            actions = []
            for _ in range(num):
                actions.append(range(0, num))

        values = np.random.rand(num_states)
        values[-1] = immediate_rewards[-1]
        if in_place:
            new_values = np.array([num_states, 1])
        while True:
            delta = -np.inf
            for i in range(0, num_states):
                if i in terminal_states:
                    continue
                tmp = []
                for j in actions[i]:
                    transition_probabilities = transition_model(j, False)
                    tmp.append((policy[i][j] * np.dot(transition_probabilities[i], (immediate_rewards + gamma * values))))
                new_v = np.sum(tmp)
                delta = max(delta, values[i] - new_v)
                if not in_place:
                    new_values[i] = new_v
                else:
                    values[i] = new_v
            if not in_place:
                values = new_values
            #print values
            if delta < theta:
                break
        return values


    @staticmethod
    def policy_iteration(initial_policy, transition_probabilities, immediate_rewards, actions, gamma=0.9,
                         theta=0.1, terminal_states=[], in_place=True):
        num_states = np.size(immediate_rewards, 0)
        if type(actions) is int:
            num = actions
            actions = []
            for _ in range(num):
                actions.append(range(0, num))

        if initial_policy is None:
            policy = []
            for state_actions in actions:
                policy.append(np.eye(1, len(state_actions)))
        else:
            policy = initial_policy
        while True:
            policy_stable = True
            values = MDP.policy_evaluation(policy, transition_probabilities, immediate_rewards, actions, gamma,
                                           theta, terminal_states, in_place)  # Policy Evaluation
            for i in range(0, num_states):
                b = np.argmax(policy[
                                  i])  # policy should have an indicator vector for each state indicating the action to be taken
                policy[i] = np.zeros(len(actions[i]))
                ind = np.argmax(
                    np.dot(transition_probabilities([i], True), (immediate_rewards[i] + gamma * values)))
                policy[i][ind] = 1
                if ind != b:
                    policy_stable = False
            if policy_stable:
                break

        return values, policy


def test():
    policy = np.array([[0.5, 0.5], [0.5, 0.5]])
    transition_probabilities = np.array(
        [np.array([[0.5, 0.5], [0.49999, 0.50001]]), np.array([[0, 1], [0, 1]])])
    transition_model = lambda a, st: transition_probabilities[a] if st else [state[a] for state in
                                                                             transition_probabilities]
    immediate_rewards = [-10, 1]
    print('Policy iteration:',
          MDP.policy_iteration(policy, transition_model, immediate_rewards, 2, theta=1e-20))
    print('Value iteration:', MDP.value_iteration(transition_model, immediate_rewards, 2, theta=1e-10), 'here')


def map_vi_test():
    map = Map.random_grid_map(4, 6)
    model = map.get_transition_model(noise=0)
    tmp = dict()
    for action in ['N', 'S', 'E', 'W']:
        for state in range(map.num_states):
            if state in tmp:
                d = tmp[state]
            else:
                d = dict()
            d[action] = model(action)[state]
            tmp[state] = d
    actions = []

    for _ in range(map.num_states):
        actions.append(['S', 'E', 'W', 'N'])
    transition_model = lambda a, st: tmp[a].values() if st else model(a)
    immediate_rewards = -np.ones(map.num_states)
    immediate_rewards[-1] = 10
    sig = lambda x: 1 / (1 + np.exp(-x))
    vals, policy = MDP.value_iteration(transition_model, immediate_rewards, actions)

    map.show(delay=2)
    map.show(sig(vals), delay=2)
    prev_state = int(np.random.choice(range(map.num_states)))
    t = actions[0]
    while True:
        action = t[np.argmax(policy[prev_state])]
        current_state = int(
            np.random.choice(range(map.num_states), p=transition_model(action, False)[prev_state]))
        map.show(sig(vals), current_state, 2)
        if current_state == map.num_states - 1:
            break
        prev_state = current_state


def map_q_test():
    num_colours = 4
    num_states = 36
    connected = False
    while not connected:
        map = Map.random_grid_map(num_colours, int(np.sqrt(36)))
        map.show(delay=0.5)
        connected = map.has_path(0, num_states - 1)
    transition_model = map.get_transition_model(noise=0)
    Q = np.random.random_sample((num_states, 4))
    mdp = MDP(init_Q_matrix=Q, epsilon=0.1)
    mdp.train_offline(method='q')
    prev_state = 0
    iteration = 0
    print('Iteration {}'.format(iteration))
    while iteration < 50:
        cardinals = ['N', 'E', 'W', 'S']
        action = [mdp.make_decision(prev_state)][0]
        card_action = cardinals[action]
        current_state = int(
            np.random.choice(range(map.num_states), p=transition_model(card_action)[prev_state]))
        if iteration in [25, 40, 49]:
            map.show(actual_state=current_state, delay=0.5)
        if current_state == map.num_states - 1:
            reward = 10
            mdp.q_matrix_update(prev_state, action, reward, current_state)
            current_state = 0
            iteration += 1
            print('Iteration {}'.format(iteration))
        else:
            reward = -1
            mdp.q_matrix_update(prev_state, action, reward, current_state)
        prev_state = current_state

def Q_test_2():
    num_colours = 2
    num_states = 9
    state_dim = num_states + 4
    num_actions = 3
    num_sensors = 3

    epsilon = 0.5
    epsilon_decay = 0.9998
    minimum_epsilon = 0.3
    dropout = 0.2

    end_state = num_states-1
    actions = ['LEFT_TURN', 'FORWARD', 'RIGHT_TURN']

    reward_vector = np.ones(num_states)*-1
    reward_vector[num_states-1] = 0

    randomize = 1
    replay = 1
    batch_size = 1

    model_num = 1

    figsize = (18.5, 8)
    gs = GridSpec(2, 2)

    fig = plt.figure(1, figsize=figsize)
    ax = fig.add_subplot(gs[:, 1])
    map = generate_random_map(num_states, num_colours, 0, num_states-1, ax, delay=1 )
    show=1
    save=1

    model = transition_model = map.get_transition_model(noise=0)

    Q = np.random.random_sample((num_states*4*8, num_actions))

    mdp = MDP(num_states=num_states, num_actions=num_actions, init_Q_matrix=Q, method='q_table', transition_model=model, immediate_rewards=reward_vector,
            lr=1e-3, epsilon=epsilon, epsilon_decay=epsilon_decay)
    save_Q = 0
    save_titles = []

    prev_state = np.zeros(state_dim)
    prev_state[num_states+1] = 1
    prev_state[0] = 1
    current_state = prev_state
    position = np.argmax(current_state[0:num_states])
    orientation = np.argmax(current_state[num_states::])
    readings = map.get_sensor_readings(position, orientation)
    prev_full_state = np.append(prev_state, readings)
    returns = []
    success = []

    for iteration in range(40000):
        print('Iteration {}'.format(iteration))
        show_graph = iteration in range(0, 50000, 100)
        tmp = []
        save_Q = iteration%1000 == 0
        for i in range(200, 50000, 200):
            tmp += list(range(i, i + 3))
        visualize = iteration in tmp

        for moves in range(31):
            action_index = mdp.make_decision(prev_full_state)
            action = actions[action_index]
            prev_position = prev_state[0:num_states]

            if action == 'FORWARD':
                card_action = action_to_cardinal(prev_state, num_states)
                position = int(np.random.choice(range(0, map.num_states), p=transition_model(card_action)[np.argmax(prev_position)]))
                current_state[num_states::] = prev_state[num_states::]
                current_state[0:num_states] = position_encoder(position, num_states)
            else:
                orientation = get_orientation(action, prev_state, num_states)
                current_state[0:num_states] = prev_state[0:num_states]
                current_state[num_states::] = orientation

            position = current_state[0:num_states]
            orientation = current_state[num_states::]
            readings = map.get_sensor_readings(np.argmax(position), orientation=np.argmax(orientation))

            if visualize:
                save_title = 'move_{}'.format(moves)
                orientation = current_state[num_states::]
                position = current_state[0:num_states]
                map.show(actual_state=np.argmax(position), orientation=orientation, delay=0.05, title='Iteration {}\n Current Score: {}\n Last Action Taken: {}\n Move: {}'.format(iteration, mdp.current_return, action, moves), show=show, save=save, save_title=save_title, fig=fig, figsize=figsize, ax=ax)
                save_titles.append(save_title)

            full_state = np.append(current_state, readings)

            if np.argmax(current_state) == end_state:
                reward = 100
                mdp.update(prev_full_state, action_index, reward, full_state, completed=1, replay=1, batch_size=20, dropout=0.5)
                connected = False
                while not connected:
                    current_state = np.zeros(state_dim)
                    current_state[num_states + np.random.choice(range(4))] = 1
                    index = np.random.choice(range(map.num_states))
                    current_state[index] = 1
                    connected = map.has_path(index, num_states - 1)

                prev_state = current_state
                prev_full_state = full_state

                success.append(1)

                break

            elif moves == 30:
                reward = -1

                mdp.update(prev_full_state, action_index, reward, full_state, completed=0, replay=1, batch_size=20, dropout=0.5)

                connected = False
                while not connected:
                    index = np.random.choice(range(map.num_states))
                    connected = map.has_path(index, num_states - 1)
                current_state = np.zeros(state_dim)
                current_state[num_states + np.random.choice(range(4))] = 1
                current_state[index] = 1

                prev_state = current_state
                prev_full_state = full_state

                success.append(0)
                break

            elif not (map.is_connected(np.argmax(prev_position), np.argmax(position), True) or (np.argmax(prev_position) == np.argmax(position) and action != 'FORWARD')):
                reward = -10

            else:
                if action == 'FORWARD':
                    reward = -1
                else:
                    reward = -10

            mdp.update(prev_full_state, action_index, reward, full_state, replay=0, batch_size=batch_size, dropout=dropout)
            prev_state = current_state
            prev_full_state = full_state

        if mdp.epsilon * mdp.epsilon_decay < minimum_epsilon:
            mdp.epsilon = minimum_epsilon
        else:
            mdp.epsilon = mdp.epsilon * mdp.epsilon_decay

        returns.append(mdp.current_return)
        mdp.current_return = 0

        if save_Q:
            np.savetxt('Q_values_{}_iteration'.format(iteration), mdp.Q_matrix, delimiter=',')

        if show_graph:
            mdp.draw_graph(returns, success, iteration, fig, gs)

        if visualize:
            mdp.create_gif(image_titles=save_titles, num_states=num_states, iteration=iteration)

        if randomize and iteration % 10 == 0:
            map = generate_random_map(num_states, num_colours, 0, num_states - 1, ax, delay=0.5)
            transition_model = map.get_transition_model(noise=0)
        save_titles = []

    mdp.log.to_csv('qrn_model.csv')

def random_agent():
    num_colours = 2
    num_states = 9
    state_dim = num_states + 4
    num_actions = 3
    num_sensors = 3

    end_state = num_states-1
    actions = ['LEFT_TURN', 'FORWARD', 'RIGHT_TURN']

    randomize = 1
    model_num = 1

    figsize = (18.5, 8)
    gs = GridSpec(2, 2)

    fig = plt.figure(1, figsize=figsize)
    ax = fig.add_subplot(gs[:, 1])
    map = generate_random_map(num_states, num_colours, 0, num_states-1, ax, delay=1 )
    transition_model = map.get_transition_model(noise=0)
    prev_state = np.zeros(state_dim)
    prev_state[num_states+1] = 1
    prev_state[0] = 1
    current_state = prev_state

    position = np.argmax(current_state[0:num_states])
    orientation = np.argmax(current_state[num_states::])
    readings = map.get_sensor_readings(position, orientation)
    prev_full_state = np.append(prev_state, readings)

    log = DataFrame(columns=['Previous State', 'Action Taken','Reward', 'Current State', 'Move', 'Episode', 'Successful Episode', 'Return'])
    prev_time = time()
    for iteration in range(60000):
        if iteration % 100 == 0:
            current_time = time()
            print('Iteration {} ({:.3f} (sec))'.format(iteration, current_time-prev_time))
            prev_time = current_time
        if iteration % 5000 == 0:
            log.to_csv('log_data_episode_{}.csv'.format(iteration), index=False)
        buffer = DataFrame(columns=['Previous State', 'Action Taken', 'Reward', 'Current State', 'Move', 'Episode'], dtype=object)

        for moves in range(51):

            action_index = np.random.choice(range(num_actions))
            action = actions[action_index]
            prev_position = prev_state[0:num_states]
            if action == 'FORWARD':
                card_action = action_to_cardinal(prev_state, num_states)
                position = int(np.random.choice(range(map.num_states), p=transition_model(card_action)[np.argmax(prev_position)]))
                current_state[num_states::] = prev_state[num_states::]
                current_state[0:num_states] = position_encoder(position, num_states)
            else:
                orientation = get_orientation(action, prev_state, num_states)
                current_state[0:num_states] = prev_state[0:num_states]
                current_state[num_states::] = orientation

            position = current_state[0:num_states]
            orientation = current_state[num_states::]
            readings = map.get_sensor_readings(np.argmax(current_state[0:num_states]), orientation=np.argmax(orientation))

            full_state = np.append(current_state, readings)
            if np.argmax(current_state) == end_state:
                reward = 1

                buffer.loc[len(buffer), :] = [prev_full_state, action_index, reward, full_state, moves, iteration]
                buffer['Successful Episode'] = np.ones(len(buffer))
                buffer['Return'] = buffer['Reward'].sum()*np.ones(len(buffer))
                log = concat([buffer, log], axis=0)

                connected = False
                while not connected:
                    current_state = np.zeros(state_dim)
                    current_state[num_states + np.random.choice(range(4))] = 1
                    index = np.random.choice(range(map.num_states))
                    current_state[index] = 1
                    connected = map.has_path(index, num_states - 1)

                prev_state = current_state.copy()
                position = current_state[0:num_states]
                orientation = current_state[num_states::]
                readings = map.get_sensor_readings(np.argmax(position), np.argmax(orientation))
                prev_full_state = np.append(prev_state, readings)
                break
                # for row in buffer.values:
                #     log.loc[len(log), :] = row.tolist()
                # break

            elif moves == 50:
                reward = -1

                buffer.loc[len(buffer), :] = [prev_full_state, action_index, reward, full_state, moves, iteration]
                buffer['Successful Episode'] = np.zeros(len(buffer))
                buffer['Return'] = buffer['Reward'].sum()*np.ones(len(buffer))
                log = concat([buffer, log], axis=0)

                connected = False
                while not connected:
                    current_state = np.zeros(state_dim)
                    current_state[num_states + np.random.choice(range(4))] = 1
                    index = np.random.choice(range(map.num_states))
                    current_state[index] = 1
                    connected = map.has_path(index, num_states - 1)
                # for row in buffer.values:
                #     log.loc[len(log), :] = row.tolist()
                # break
                position = current_state[0:num_states]
                orientation = current_state[num_states::]
                prev_state = current_state.copy()
                readings = map.get_sensor_readings(np.argmax(position), np.argmax(orientation))
                prev_full_state = np.append(prev_state, readings)
                break

            elif np.argmax(prev_position) == np.argmax(position) and (action == 'FORWARD'):
                reward = -0.2

            elif action == 'FORWARD':
                pos_ind = np.argmax(position)
                length = np.sqrt(num_states)
                reward = -float(length-(pos_ind % length) + length-(pos_ind/length))/(20*length)
            else:
                reward = -0.1


            buffer.loc[len(buffer), :] = [prev_full_state, action_index, reward, full_state, moves, iteration]
            prev_state = current_state.copy()
            prev_full_state = full_state.copy()

        if randomize and iteration % 5 == 0:
            map = generate_random_map(num_states, num_colours, np.argmax(position), num_states - 1, ax, delay=0.5)
            transition_model = map.get_transition_model(noise=0)

    log.to_csv('random_model.csv', index=False)

def readings_encoder(readings):
    tmp = np.zeros(8)
    index = int(readings[0] + 2*readings[1] + 4*readings[2])
    tmp[index] = 1
    return np.array(tmp, dtype=int)

def action_encoder(action, encoder_type=0):
    if encoder_type == 0:
        return action == np.array(['N', 'E', 'W', 'S'])
    else:
        return action == np.array(['LEFT_TURN', 'FORWARD', 'RIGHT_TURN'])


def position_encoder(position, num_states):
    tmp = np.zeros(num_states)
    tmp[position] = 1
    return tmp


def action_to_cardinal(prev_state, num_states):
    orientation = prev_state[num_states::]

    if orientation[0]:
        return 'W'
    if orientation[1]:
        return 'N'
    if orientation[2]:
        return 'E'
    if orientation[3]:
        return 'S'


def get_orientation(action, prev_state, num_states):
    orientation = prev_state[num_states::]
    new_orientation = np.zeros(4)
    i = np.argmax(orientation)
    if action == 'LEFT_TURN':
        new_orientation[(i-1) % 4] = 1
    elif action == 'RIGHT_TURN':
        new_orientation[(i+1) % 4] = 1
    return new_orientation


def generate_random_map(num_states, num_colours, start, end, ax, delay=1.0):
    connected = False
    while not connected:
        map = Map.random_grid_map(num_colours, int(np.sqrt(num_states)))
        map.show(delay=delay,ax=ax, show=0)
        ax.cla()
        connected = map.has_path(start, end)
    return map





if __name__ == '__main__':
    random_agent()
