import numpy as np
import tensorflow as tf
from keras import backend as K

class network_operations:
    """Compilation of functions used deal with Q-networks, policy-networks and actor-critic networks"""

    def __init__(self,sess, action_dimension, lr=1.e-3,learning_method='SARSA', policy_type='softmax', use_target_models=0, tau=1.e-3, state_formatter=lambda x, y: x, action_formatter=lambda x: x, batch_state_formatter=lambda x, y: x):
        self.SESSION = sess
        self.ACTION_DIM = action_dimension
        self.LR = lr
        self.LEARNING_METHOD = learning_method.upper()
        self.POLICY = policy_type.lower()
        self.TARGET_MODEL = use_target_models
        self.TAU = tau
        self.state_formatter =  state_formatter
        self.action_formatter = action_formatter
        self.batch_state_formatter = batch_state_formatter
        K.set_session(sess)

    def action_encoder(self, action_index):
        encoded_action = np.zeros(self.ACTION_DIM)
        encoded_action[action_index] = 1
        return encoded_action

    # Policy networks
    def get_policy_update_operation(self, policy_network):
        policy = policy_network.output
        R = tf.placeholder('float') # Return holder
        loss = tf.nn.log_softmax(policy*R)
        operation = tf.train.AdamOptimizer(self.LR).minimize(loss=loss) # update operation
        return operation, R

    # Actor-Critic networks
    def get_actor_update_operation(self, actor_model):
        policy = actor_model.output
        action_gradients = tf.placeholder('float', shape = [None, self.ACTION_DIM])

        weights = actor_model.trainable_weights
        gradient_parameters = tf.gradients(policy, weights, -action_gradients)
        grads = zip(gradient_parameters, weights)

        operation = tf.train.AdamOptimizer(self.LR).apply_gradients(grads)
        return operation, action_gradients

    def get_critic_gradient_operation(self, critic_model):
        """
        Gradient operation used to update the actor network
        :param critic_model: critic neural-network
        :return tensorflow operation:
        """
        Q_function = critic_model.output
        actions = critic_model.inputs[2]
        action_gradient_op = tf.gradients(Q_function, actions)
        return action_gradient_op

    def get_critic_gradients(self, gradient_op, critic_model, inputs):
        """
        Runs critic's gradient operation to get gradients to update actor network
        :param gradient_op: tensorflow operation for calculating critic gradients
        :param critic_model: critic neural network
        :param inputs: list of inputs to feed the critic model
        :return:
        """
        critic_inputs = critic_model.inputs
        d = {}
        for i, ci in enumerate(critic_inputs):
            d[ci] = inputs[i]
        return self.SESSION.run([gradient_op],feed_dict=d)

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

    def batch_train_actor_critic_model(self, models, episodes, gamma, tf_holders, iterations, batch_size, vanilla_actor):
        sess = self.SESSION
        actor_model = models[0]
        critic_model = models[1]

        for iteration in range(iterations):
            print('Iteration: {}'.format(iteration))
            targets = []
            states = []
            actions = []
            deltas = []
            for _, frame in episodes.sample(batch_size).iterrows():
                raw_state, action, reward, raw_next_state = frame[0], frame[1], frame[2], frame[3]
                state = self.state_formatter(raw_state, action)

                if self.TARGET_MODEL:
                    q = models[3].predict(state, batch_size=1).flatten()
                else:
                    q = critic_model.predict(state, batch_size=1).flatten()

                if not vanilla_actor:
                    next_state = self.state_formatter(raw_next_state)
                    if self.TARGET_MODEL:
                        next_policy = models[2].predict(next_state, batch_size=1).flatten()
                    else:
                        next_policy = actor_model.predict(next_state, batch_size=1).flatten()
                else:
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
                            if not vanilla_actor:
                                next_action = self.action_encoder(np.random.choice(self.ACTION_DIM, p=next_policy))
                            else:
                                next_action = self.action_encoder(np.random.choice(self.ACTION_DIM, p=next_policy))

                    elif self.LEARNING_METHOD == 'Q-learning':
                        indx = np.argmax(next_policy)
                        next_action = self.action_encoder(indx)

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
            gradients = self.get_critic_gradients(tf_holders[2], critic_model, batch_states)
            gradients = np.squeeze(gradients)
            gradients = np.reshape(gradients, (-1, self.ACTION_DIM))

            if vanilla_actor:
                batch_states=self.batch_state_formatter(states)
                sess.run([tf_holders[1]], feed_dict={actor_model.input: batch_states, tf_holders[0]: gradients})
            else:
                batch_states=self.batch_state_formatter(states)
                d = dict(zip(actor_model.inputs, batch_states))
                d[tf_holders[0]] = gradients
                sess.run([tf_holders[1]], feed_dict=d)

            if self.TARGET_MODEL:
                self.update_target_models(models[0], models[2])
                self.update_target_models(models[1], models[3])

    def batch_train_policy_model(self, models, episodes, gamma, tf_holders, iterations, batch_size, vanilla_actor):
        sess = self.SESSION
        policy_model = models[0]

        for iteration in range(iterations):
            print('Iteration: {}'.format(iteration))
            returns = []
            policies = []

            for _, frame in episodes.sample(batch_size).iterrows():
                raw_state, action, R = frame['States'], frame['Actions'], frame['Returns']
                state = self.state_formatter(raw_state, action)

                if self.TARGET_MODEL:
                    policy = models[1].predict(state, batch_size=1).flatten()
                else:
                    policy = policy_model.predict(state, batch_size=1).flatten()
                policies.append(policy)
                returns.append(R)

            d = dict()
            d[tf_holders[2]] = np.reshape(returns, (-1, len(returns)))
            d[tf_holders[0]] = policies
            sess.run([tf_holders[1]], feed_dict=d)
            if self.TARGET_MODEL:
                self.update_target_models(models[0], models[1])
