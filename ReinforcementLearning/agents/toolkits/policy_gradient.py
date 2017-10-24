import numpy as np
import tensorflow as tf
from keras import backend as K

class Policy_Gradient:
    """Compilation of functions used deal with Q-networks, policy-networks and actor-critic networks"""

    def __init__(self, sess, action_dimension, gamma=0.9, lr=1.e-3, beta=0,learning_method='SARSA', policy_type='softmax', use_target_models=0, tau=1.e-3, state_formatter=lambda x, y: x, action_formatter=None, batch_state_formatter=lambda x, y: x):
        self.SESSION = sess
        self.ACTION_DIM = action_dimension
        self.LR = lr
        self.BETA = beta
        self.LEARNING_METHOD = learning_method.upper()
        self.POLICY = policy_type.lower()
        self.TARGET_MODEL = use_target_models
        self.TAU = tau
        self.GAMMA = gamma
        self.state_formatter = state_formatter
        self.action_formatter = self.action_encoder if action_formatter is None else action_formatter
        self.batch_state_formatter = batch_state_formatter
        K.set_session(sess)

    @staticmethod
    def entropy(policy):
        return -tf.reduce_sum(policy*tf.nn.log_softmax(policy))

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


    # Monte-Carlo policy network functions
    def get_policy_update_operation_MC(self, policy_network):
        policy = policy_network.output
        R = tf.placeholder('float')  # Return holder
        loss = tf.nn.log_softmax(policy * R)- self.BETA*self.entropy(policy)
        operation = tf.train.AdamOptimizer(self.LR).minimize(loss=loss)  # update operation
        return operation, R


    def batch_train_policy_model_MC(self, models, episodes, gamma, tf_holders, iterations, batch_size, vanilla_actor):
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
