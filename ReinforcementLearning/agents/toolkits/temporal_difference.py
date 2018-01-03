import numpy as np

# TD(0)

class Q_Table:
    def __init__(self, init_table, lr=1e-3, gamma=1e-3):
        self.q_table = init_table
        self.lr = lr
        self.gamma = gamma

    def predict(self, state_index):
        return self.q_table[state_index]

    def update(self, state, action, reward, next_state):
        self.q_table[state][action] += self.lr * (reward + self.gamma * max(self.q_table[next_state]) - self.q_table[state][action])

def q_learning(q_table, episodes, gamma, lr, iterations, state_index_func, policy, policy_learning_algorithm,
               epsilon, epsilon_decay, epsilon_minimum, heat, heat_decay, heat_minimum, random_state):
    for _, frame in episodes.sample(frac=iterations, random_state=random_state).iterrows():
        state, action, reward, next_state = frame.iloc[:, 0], frame.iloc[:, 1], frame.iloc[:, 2], frame.iloc[:, 3]
        state_index = (state)
        next_state_index = state_index_func(next_state)
        future_q_values = q_table[next_state_index]
        if policy_learning_algorithm == 'Q-Learning':
            target = gamma * np.max(future_q_values)
        elif policy_learning_algorithm == 'SARSA':
            if policy == 'softmax':
                p = np.exp(future_q_values) / sum(np.exp(future_q_values))
                target = gamma * np.dot(future_q_values, p)
                heat = min(heat_minimum, heat * heat_decay)
            elif policy == 'epsilon-greedy':
                if random_state.rand() < epsilon:
                    target = gamma * random_state.choice(future_q_values)
                else:
                    target = gamma * np.max(future_q_values)
                epsilon = min(epsilon_minimum, epsilon * epsilon_decay)
        q_table[state_index, action] += lr * (reward + target - q_table[state_index, action])
    return q_table
