import numpy as np
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import InputLayer
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from pandas import DataFrame, concat

from agents.advanced_networks import network_toolkit


class MDP:
    """Basic implementation of a finite Markov Decision Process    
    """

    def __init__(self, num_states=None, num_actions=None, gamma=0.9,lr=1e-3, method ='q-network', policy_type='epsilon-greedy', epsilon = 0.1, epsilon_decay=0.995, minimum_epsilon=0.1,heat=0.1, heat_decay=0.995, minimum_heat=0.1,
                 transition_model=None, immediate_rewards=None, value_function=None, function_type=0, policy=None, init_Q_matrix=None, q_model=None, v_model=None, target_models=[], tau=1e-3, actor_model=None, critic_model=None, network_type='Q', folder_name='Default', log=None, state_formatter=None, random_state=None,sess=None, **kwargs ):
        """
        Initializer for MDP agents

        Parameters
        ----------

        num_states:        int - state space size
        num_actions:       int - action space size
        gamma:             float - discount parameter used in updates
        lr:                float - learning rate
        method:            string - method used to determine policy (Case insensitive). Options ['vi', 'pi', 'q-table', 'q-network', 'policy-network']
        epsilon:           float - epsilon used in epsilon-greedy policy
        epsilon_decay:     float - epsilon is discounted by this amount after each episode
        minimum_epsilon:   float - smallest value epsilon is allowed to take
        heat:              float - Tau used in blotzman/ softmax policy
        heat_decay:        float - Tau is discounted by this amount after each episode
        minimum_heat:      float - smallest value Tau is allowed to take
        transition_model:  f: int -> int [state_dim, state_dim] - function that returns float [state_dim, state_dim] (transition probability from each state to another state) when given action
        immediate_rewards: f: int x int -> float - function that returns float (reward) when given state index and action index
        value_funtion:     f: int -> float - function that returns value of state when given state index
        function_type:     int - Index indicating function type
        policy:            f: int -> float [num_actions, 1] - function that returns probabilities of choosing an action when given a state index
        init_Q_matrix:     float[state_dim, num_actions] - initial Q matrix
        q-model:           keras neural-network -  approximates the Q matrix
        v-model:           keras neural-network -  approximates the V function
        target_models:     list of keras neural-network(s)- used to update networks more slowly
        tau:               float - learning rate used in updating target networks
        network_type:      String - Type of neural-network (Case insensitive). Options ['Q', 'V']
        log:               DataFrame - Dataframe used to store episode info
        folder:            String- name of the folder to store data saved from agents
        state_formatter:   Function that formats states
        random_state:      Random state used for all random processes
        sess:              Tensorflow session
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
        self.heat=heat
        self.minimum_epsilon = minimum_epsilon
        self.minimum_heat = minimum_heat
        self.gamma = gamma
        self.lr = lr
        self.epsilon_decay = epsilon_decay
        self.heat_decay = heat_decay
        self.actor_model=actor_model
        self.critic_model = critic_model
        self.target_models = target_models
        if q_model is None and v_model is None:
                self.model = self._build_NN(**kwargs) #Q-Neural Network
        elif q_model is not None:
            self.model = q_model
        else:
            self.model = v_model
        self.network_type = network_type.capitalize()
        self.method = method.lower()
        self.current_return = 0
        self.buffer = DataFrame(columns=['Previous State', 'Action Taken', 'Current State', 'Reward', 'Completed'])
        if log is None:
            self.log = DataFrame(columns=['Previous State', 'Action Taken', 'Current State', 'Reward', 'Completed', 'Total Return'])
        else:
            self.log = log
        self.foldername = folder_name
        f=lambda x:x
        self.state_formatter = f if state_formatter is None else state_formatter
        self.random_state = np.random if random_state is None else random_state
        self.toolkit = network_toolkit(sess=sess, action_dimension=num_actions, use_target_models=target_models!=[], tau=tau)

    def train_offline(self, transition_model=None, immediate_returns=None, method='vi', gamma=0.9, theta=1e-10, alpha=0.8, terminal_states=[], in_place=True, initial_policy=None, episodes=None, iterations=0, policy_learning_algorithm='Q-Learning'):
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

        if method == 'q-table':
            self.num_states, self.num_actions = np.size(self.Q_matrix, 0), np.size(self.Q_matrix, 1)
            self.function_type = 1
            self.alpha = alpha
            self.gamma = gamma
            self.Q_matrix = self.q_learning_offline(self.Q_matrix, episodes, self.gamma, self.alpha, iterations, self.state_formatter, self.policy_type, policy_learning_algorithm, self.epsilon, self.epsilon_decay, self.minimum_epsilon, self.heat, self.heat_decay, self.minimum_heat, self.random_state)

    def q_matrix_update(self, state, action, reward, next_state):
        self.Q_matrix[state][action] += self.lr * (reward + self.gamma * max(self.Q_matrix[next_state]) - self.Q_matrix[state][action])

    def _build_NN(self, num_inputs=None, num_hidden_neurons=10, num_hidden_layers=1, num_output_neurons=4, dropout_rate=0.2, lr=0.001, recurrent_layer=0, num_rnn_units=128, lookback=0, convolutional_layer=0, filters=16, kernel_size=2, l2_reg=0.7):
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


    def make_decision(self, state):
        state = self.state_formatter(state)

        if self.policy_type =='random':
            return self.random_state.choice(range(self.num_actions))

        elif self.policy_type == 'epsilon-greedy':

            if self.random_state.rand() <= self.epsilon:
                return self.random_state.choice(range(self.num_actions))

            if self.method in ['q-network', 'policy-network']:
                if self.target_models is not []:
                    q_values = np.array(self.target_models[0].predict(state)).flatten()
                else:
                    q_values = np.array(self.model.predict(state)).flatten()

            elif self.method == 'q-table':
                q_values = self.Q_matrix[state]

            elif self.method == 'actor-critic':
                if self.target_models is not []:
                     q_values = np.array(self.target_models[0].predict(state)).flatten()
                else:
                     q_values = np.array(self.actor_model.predict(state)).flatten()
            return np.argmax(q_values)

        elif self.policy_type == 'softmax':
            if self.method in ['policy-network', 'actor-critc']:
                p = np.array(self.model.predict(state)).flatten()

            elif self.method == 'q-network':
                if self.target_models is not []:
                    tmp = np.array(self.target_models[0].predict(state)).flatten()
                else:
                    tmp = np.exp(np.array(self.model.predict(state).flatten()))
                p = tmp / np.sum(tmp)
                
            elif self.method == 'q-table':
                tmp = np.exp(np.array(self.Q_matrix[state]))
                p = (tmp/np.sum(tmp)).flatten()
            elif self.method == 'actor-critic':
                if self.target_models is not []:
                    p = np.array(self.target_models[0].predict(state)).flatten()
                else:
                    p = np.array(self.actor_model.predict(state)).flatten()
            print(p)
            return self.random_state.choice(range(len(p)), p=p)


    def update(self, state, action, reward, next_state, done=False, log=True, replay=True, batch_size=1, minibatch_size=1, dropout=0.2):
        raw_state = state
        raw_next_state = next_state
        self.current_return += reward
        state = self.state_formatter(raw_state)
        next_state = self.state_formatter(raw_next_state)
        if log:
            self.buffer.loc[len(self.buffer), :] = [raw_state, action, raw_next_state, reward, done]
            if done:
                self.buffer['Total Return'] = self.buffer['Reward'].sum()
                self.log = concat([self.buffer, self.log], axis=0)
                self.log = self.log.iloc[0: min(len(self.log), 10000), :]
                self.buffer = DataFrame(columns=['Previous State', 'Action Taken', 'Current State', 'Reward', 'Completed'])
        if done:
            self.epsilon = np.max([self.epsilon*self.epsilon_decay, self.minimum_epsilon])

        if self.random_state.random_sample() < dropout:
            return

        if self.method == 'q-table':
            if self.random_state.random_sample() < replay:
                self.train_offline(episodes=self.log, method=self.method)
            else:
                self.q_matrix_update(self.state_formatter(state), action, reward, self.state_formatter(next_state))

        elif self.method == 'q-network':
            
            if self.random_state.random_sample() < replay:
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
                self.model.fit(states[0], np.array(targets)[0], batch_size=minibatch_size, epochs=1, verbose=0)

            else:
                future_q = np.max(self.model.predict(next_state, batch_size=1).flatten())
                q = self.model.predict(state, batch_size=1).flatten()
                target = reward + self.gamma*future_q
                q[action] = target
                q = np.reshape(q, (-1, len(q)))
                self.model.fit(state, q, batch_size=1, epochs=1, verbose=0)
            if self.target_models is not []:
                self.toolkit.update_target_models(self.model, self.target_models[0])

        elif self.method == 'actor-critic':
            if self.random_state.rand() < replay:
                episodes = self.log
            else:
                episodes = self.buffer.iloc[0, [0,1,2,3]]
            if done:
                self.toolkit.batch_train_actor_critic_model(models=[self.actor_model, self.critic_model]+self.target_models, episodes=episodes, iterations=1, batch_size=batch_size )

        elif self.method == 'policy-network':
            pass


    @staticmethod
    def value_iteration(transition_probabilities, immediate_rewards, actions, gamma=0.9, theta=0.1,
                        terminal_states=[], in_place=True, initial_policy=None):
        """

        :param transition_probabilities:
        :param immediate_rewards:
        :param actions:
        :param gamma:
        :param theta:
        :param terminal_states:
        :param in_place:
        :param initial_policy:
        :return:

        >>> import numpy as np
        >>> policy = np.array([[0.5, 0.5], [0.5, 0.5]])
        >>> transition_probabilities = np.array([np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0, 1], [0, 1]])])
        >>> transition_model = lambda a, st: transition_probabilities[a] if st else [state[a] for state in transition_probabilities]
        >>> immediate_rewards = [-10, 1]
        >>> print('Values:{val[0]}, Policy:{val[1]}'.format(val = MDP.value_iteration(transition_model, immediate_rewards, 2, theta=1e-10)))
            Values:[-5.5112429  4.0951   ], Policy:[array([ 1.,  0.]), array([ 1.,  0.])]

        """
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
        """

        :param initial_policy:
        :param transition_probabilities:
        :param immediate_rewards:
        :param actions:
        :param gamma:
        :param theta:
        :param terminal_states:
        :param in_place:
        :return: values, policy

        >>> import numpy as np
        >>> policy = np.array([[0.5, 0.5], [0.5, 0.5]])
        >>> transition_probabilities = np.array([np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0, 1], [0, 1]])])
        >>> transition_model = lambda a, st: transition_probabilities[a] if st else [state[a] for state in transition_probabilities]
        >>> immediate_rewards = [-10, 1]
        >>> print('Values:{val[0]}, Policy:{val[1]}'.format(val = MDP.policy_iteration(policy, transition_model, immediate_rewards, 2, theta=1e-10)))
            Values:[-5.52905259  4.0951    ], Policy:[[ 1.  0.] [ 1.  0.]]
        """
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
                b = np.argmax(policy[i])  # policy should have an indicator vector for each state indicating the action to be taken
                policy[i] = np.zeros(len(actions[i]))
                ind = np.argmax(
                    np.dot(transition_probabilities([i], True), (immediate_rewards[i] + gamma * values)))
                policy[i][ind] = 1
                if ind != b:
                    policy_stable = False
            if policy_stable:
                break

        return values, policy
    @staticmethod
    def q_learning_offline(q_table, episodes,gamma, lr, iterations, state_index_func, policy, policy_learning_algorithm,epsilon, epsilon_decay, epsilon_minimum, heat, heat_decay, heat_minimum, random_state):
        for _, frame in episodes.sample(frac=iterations, random_state=random_state).iterrows():
            state, action, reward, next_state = frame.iloc[:, 0], frame.iloc[:, 1], frame.iloc[:, 2], frame.iloc[:, 3]
            state_index = (state)
            next_state_index = state_index_func(next_state)
            future_q_values = q_table[next_state_index]
            if policy_learning_algorithm == 'Q-Learning':
                target = gamma*np.max(future_q_values)
            elif policy_learning_algorithm == 'SARSA':
                if policy=='softmax':
                    p = np.exp(future_q_values)/sum(np.exp(future_q_values))
                    target = gamma*np.dot(future_q_values, p)
                    heat = min(heat_minimum, heat*heat_decay)
                elif policy == 'epsilon-greedy':
                    if random_state.rand()< epsilon:
                        target = gamma*random_state.choice(future_q_values)
                    else:
                        target = gamma*np.max(future_q_values)
                    epsilon = min(epsilon_minimum, epsilon*epsilon_decay)
            q_table[state_index, action] += lr*(reward + target - q_table[state_index, action])
        return q_table

if __name__ == '__main__':
    pass
