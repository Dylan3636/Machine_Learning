#setting seed
import os
os.environ['PYTHONHASHSEED'] = '0'
import numpy as np
SEED = 5  # 15,485,863
np.random.seed(SEED)
import random
random.seed(SEED)
RANDOM_STATE = np.random.RandomState(seed=SEED)

import tensorflow as tf
from keras import backend as K

class Actor_Critic:
    """Compilation of functions used deal with Q-networks, policy-networks and actor-critic networks"""

    def __init__(self, sess, action_dimension, gamma=0.9, lr=1.e-3,learning_method='SARSA', policy_type='softmax', use_target_models=0, tau=1.e-3, state_formatter=lambda x, y: x, action_formatter=None, batch_state_formatter=lambda x, y: x):
        self.SESSION = sess
        self.ACTION_DIM = action_dimension
        self.LR = lr
        self.LEARNING_METHOD = learning_method.upper()
        self.POLICY = policy_type.lower()
        self.TARGET_MODEL = use_target_models
        self.TAU = tau
        self.GAMMA = gamma
        self.state_formatter = state_formatter
        self.action_formatter = self.action_encoder if action_formatter is None else action_formatter
        self.batch_state_formatter = batch_state_formatter
        K.set_session(sess)


    def action_encoder(self, action_index):
        encoded_action = np.zeros(self.ACTION_DIM)
        encoded_action[action_index] = 1
        return encoded_action

    def set_formatters(self,  state_formatter=lambda x, y: x, action_formatter=lambda x: x, batch_state_formatter=lambda x, y: x):
        self.state_formatter = state_formatter
        self.action_formatter = self.action_encoder if action_formatter is None else action_formatter
        self.batch_state_formatter = batch_state_formatter

    def entropy(policy):
        return -tf.reduce_sum(policy * tf.nn.log_softmax(policy))


    # Actor-Critic networks
    def get_actor_update_operation(self, actor_model):
        policy = actor_model.output
        weights = actor_model.trainable_weights

        critic_gradients = tf.placeholder('float', shape=[None, self.ACTION_DIM])
        loss = -tf.nn.log_softmax(policy) * critic_gradients - self.BETA * self.entropy(policy)
        gradients = tf.gradients(ys=loss, xs=weights)
        grads = zip(gradients, weights)
        operation = tf.train.AdamOptimizer(LR).apply_gradients(grads)

        return operation, critic_gradients

    def get_actor_update_operation_ddpg(self, actor_model):
        policy = actor_model.output
        action_gradients = tf.placeholder('float', shape = [None, self.ACTION_DIM])

        weights = actor_model.trainable_weights
        gradients = tf.gradients(tf.nn.log_softmax(policy), weights, -action_gradients)
        grads = zip(gradients, weights)

        operation = tf.train.AdamOptimizer(self.LR).apply_gradients(grads)
        return operation, action_gradients

    def set_actor_update_op(self, actor_model=None, actor_update_op=None, critic_gradient_holder=None):
        if actor_update_op is not None and critic_gradient_holder is not None:
            self.actor_update_op=actor_update_op
            self.critic_gradient_holder = critic_gradient_holder
        else:
            self.actor_update_op, self.critic_gradient_holder = self.get_actor_update_operation_ddpg(actor_model=actor_model)


    def get_critic_gradient_operation(self, critic_model, actions_index=-1):
        """
        Gradient operation used to update the actor network
        :param critic_model: critic neural-network
        :return tensorflow operation:
        """
        Q_function = critic_model.output
        actions = critic_model.inputs[actions_index]
        action_gradient_op = tf.gradients(Q_function, actions)
        return action_gradient_op

    def set_critic_gradient_operation(self, critic_model=None, critic_gradient_op=None):
        if critic_gradient_op is not None:
            self.critic_gradient_op = critic_gradient_op
        elif critic_model is not None:
            self.critic_gradient_op = self.get_critic_gradient_operation(critic_model)


    def get_critic_gradients(self, critic_model, inputs):
        """
        Runs critic's gradient operation to get gradients to update actor network
        :param critic_model: critic neural network
        :param inputs: list of inputs to feed the critic model
        :return:
        """
        gradient_op = self.critic_gradient_op
        critic_inputs = critic_model.inputs
        d = dict()
        for i, ci in enumerate(critic_inputs):
            try:
                d[ci] = np.squeeze(inputs[i], axis=1)
            except:
                d[ci] = inputs[i]
        return self.SESSION.run(gradient_op, feed_dict=d)

    # Target networks
    def update_target_models(self, model, target_model):
        """
        Update function to slowly update target networks
        :param target_model:
        :param model:
        :return:
        """

        # Training target model
        weights = model.get_weights()
        target_weights = target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.TAU * weights[i] + (1 - self.TAU) * target_weights[i]
        target_model.set_weights(target_weights)

    def batch_train_actor_critic_model(self, models, episodes, iterations, batch_size):
        sess = self.SESSION
        gamma = self.GAMMA
        actor_model = models[0]
        critic_model = models[1]

        for iteration in range(iterations):
            targets = []
            states = []
            actions = []
            deltas = []
            for _, frame in episodes.sample(batch_size, random_state=RANDOM_STATE).iterrows():
                frame = frame.values
                raw_state, action, raw_next_state, reward = frame[0], frame[1], frame[2], frame[3]

                state = self.state_formatter(raw_state, action)

                if self.TARGET_MODEL:
                    q = models[3].predict(state, batch_size=1).flatten()
                else:
                    q = critic_model.predict(state, batch_size=1).flatten()
                next_state = self.state_formatter(raw_next_state)
                if self.TARGET_MODEL:
                    next_policy = models[2].predict(next_state, batch_size=1).flatten()
                else:
                    next_policy = actor_model.predict(next_state, batch_size=1).flatten()

                if reward == 1:
                    target = reward
                else:
                    # Using SARSA
                    if self.LEARNING_METHOD == 'SARSA':

                        # Epsilon-Greedy Policy
                        if self.POLICY == 'epsilon-greedy':
                            indx = np.argmax(next_policy)
                            if np.random.rand() < 0.5:
                                next_action = self.action_encoder(np.random.choice(self.ACTION_DIM))
                            else:
                                next_action = self.action_encoder(indx)

                        # Softmax policy
                        elif self.POLICY == 'softmax':
                                next_action = self.action_encoder(np.random.choice(self.ACTION_DIM, p=next_policy))

                    elif self.LEARNING_METHOD == 'Q-learning':
                        indx = np.argmax(next_policy)
                        next_action = self.action_encoder(int(indx))

                    if self.TARGET_MODEL:
                        future_q = models[3].predict(self.state_formatter(raw_next_state, next_action), batch_size=batch_size).flatten()
                    else:
                        future_q = critic_model.predict(self.state_formatter(raw_next_state, next_action), batch_size=batch_size).flatten()
                    target = reward + gamma * future_q

                actions.append(action)
                deltas.append(target-q)
                targets.append(target)
                states.append(raw_state)

            batch_states=self.batch_state_formatter(states, actions)
            critic_model.train_on_batch(batch_states, np.array(targets))
            gradients = self.get_critic_gradients(critic_model, batch_states)
            gradients = np.squeeze(gradients)
            gradients = np.reshape(gradients, (batch_size, self.ACTION_DIM))

            batch_states=self.batch_state_formatter(states)
            d = dict(zip(actor_model.inputs, batch_states))
            d[self.critic_gradient_holder] = gradients
            sess.run([self.actor_update_op], feed_dict=d)


