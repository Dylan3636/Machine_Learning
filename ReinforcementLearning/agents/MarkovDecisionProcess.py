import numpy as np
from pandas import DataFrame, concat
from agents.toolkits import dynamic_programming, temporal_difference
from agents.toolkits.temporal_difference import Q_Table

class MDP:
    """Basic implementation of a finite Markov Decision Process Agent
    """

    def __init__(self, num_states=None, num_actions=None, gamma=0.9, lr=1e-3, method='q-network', policy_type='epsilon-greedy', epsilon = 0.1, epsilon_decay=0.995, minimum_epsilon=0.1,heat=0.1, heat_decay=0.995, minimum_heat=0.1,alpha=0,
                 transition_model=None, immediate_rewards=None, value_function=None, function_type=0, policy=None, init_Q_matrix=None, init_weights=None, q_model=None, v_model=None, target_models=[], tau=1e-3,
                 actor_model=None, critic_model=None, network_type='Q', folder_name='Default', log=None, state_formatter=None, random_state=None,sess=None):
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
        init_weights:      float[state_dim+1, num_actions] - initial weights for linear model
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
        self.linear_weights = init_weights
        self.epsilon = epsilon
        self.heat=heat
        self.minimum_epsilon = minimum_epsilon
        self.minimum_heat = minimum_heat
        self.gamma = gamma
        self.lr = lr
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.heat_decay = heat_decay
        self.actor_model=actor_model
        self.critic_model = critic_model
        self.target_models = target_models
        self.method = method.lower()
        if self.method =='q-table':
            self.model = Q_Table(init_table=init_Q_matrix, lr=lr, gamma=gamma)
            self.td_toolkit = temporal_difference
        elif self.method == 'q-network':
            self.model = q_model
        elif self == 'v-network':
            self.model = v_model
        elif self.method =='actor-critic':
            from agents.toolkits.actor_critic import Actor_Critic
            self.ac_toolkit = Actor_Critic(sess=sess, action_dimension=num_actions, use_target_models=target_models != [], tau=tau)
        elif self.method in ['vi', 'pi']:
            self.dp_toolkit = dynamic_programming
        self.network_type = network_type.capitalize()
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

    def train_offline(self, transition_model=None, immediate_returns=None, method='vi', gamma=0.9, theta=1e-10, alpha=0.8, terminal_states=[], in_place=True, initial_policy=None, episodes=None, iterations=0, policy_learning_algorithm='Q-Learning'):
        method = method.strip().lower()
        if method in ['vi', 'pi' ] and (transition_model is None or immediate_returns is None):
            raise Exception(
                'Transition probabilities and immediate returns cannot be none when using value iteration or policy iteration.'
            )
        if method == 'vi':
            self.function_type = 0
            self.value_function, self.policy = self.dp_toolkit.value_iteration(transition_model, immediate_returns, gamma, theta, terminal_states, in_place)

        if method == 'pi':
            self.function_type = 0
            self.value_function, self.policy = self.dp_toolkit.policy_iteration(initial_policy, transition_model, immediate_returns, gamma, theta, terminal_states, in_place)

        if method == 'q-table':
            self.num_states, self.num_actions = np.size(self.Q_matrix, 0), np.size(self.Q_matrix, 1)
            self.function_type = 1
            self.alpha = alpha
            self.gamma = gamma
            self.Q_matrix = self.td_toolkit.q_learning(self.Q_matrix, episodes, self.gamma, self.alpha, iterations, self.state_formatter, self.policy_type, policy_learning_algorithm, self.epsilon, self.epsilon_decay, self.minimum_epsilon, self.heat, self.heat_decay, self.minimum_heat, self.random_state)

    def q_matrix_update(self, state, action, reward, next_state):
        self.Q_matrix[state][action] += self.lr * (reward + self.gamma * max(self.Q_matrix[next_state]) - self.Q_matrix[state][action])

    def make_decision(self, state):
        state = self.state_formatter(state)

        if self.policy_type =='random':
            return self.random_state.choice(range(self.num_actions))

        elif self.policy_type == 'epsilon-greedy':

            if self.random_state.rand() <= self.epsilon:
                return self.random_state.choice(range(self.num_actions))

            if self.method in ['q-network', 'policy-network']:
                if self.target_models != []:
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
            elif self.method == 'q-linear':
                q_values = np.dot(self.linear_weights.T, state)
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
        elif self.method == 'q-linear':
            episodes = self.buffer if len(self.log) == 0 else self.log
            if self.random_state.random_sample() < replay:
                updates = []
                states = []
                for _, frame in episodes.sample(min(len(episodes), batch_size)).iterrows():
                    state, action, next_state, reward, done = frame.iloc(axis=1)[0:5]
                    state = self.state_formatter(state)
                    next_state = self.state_formatter(next_state)

                    q = np.dot(self.linear_weights.T, state)
                    if done:
                        target = reward
                    else:
                        future_q = np.max(np.dot(self.linear_weights.T, next_state))
                        target = reward + self.gamma * future_q
                    update = target-q[action]
                    updates.append(update)
                    states.append(state)
                #print(self.linear_weights, np.sum(np.dot(updates, states)))
                self.linear_weights[:, action] += self.lr*(np.dot(updates, states))

            else:
                future_q = np.max(np.dot(self.linear_weights.T, next_state))
                q = np.dot(self.linear_weights.T, state)
                target = reward + self.gamma*future_q
                update = target-q[action]
                self.linear_weights[:, action] += self.lr*(update)

        elif self.method == 'q-network':
            
            if self.random_state.random_sample() < replay:
                episodes = self.buffer if len(self.log)==0 else self.log
                targets = []
                states = []

                for _, frame in episodes.sample(len(episodes) if len(episodes) < batch_size else batch_size).iterrows():
                    state, action, next_state, reward, done = frame.iloc(axis=1)[0:5]
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

            if self.target_models != []:
                self.ac_toolkit.update_target_models(self.model, self.target_models[0])

        elif self.method == 'actor-critic':
            if self.random_state.rand() < replay:
                episodes = self.log
            else:
                episodes = self.buffer.iloc[0, [0, 1, 2, 3]]
            if done:
                self.ac_toolkit.batch_train_actor_critic_model(models=[self.actor_model, self.critic_model] + self.target_models, episodes=episodes, iterations=1, batch_size=batch_size)

        elif self.method == 'policy-network':
            pass



if __name__ == '__main__':
    pass
