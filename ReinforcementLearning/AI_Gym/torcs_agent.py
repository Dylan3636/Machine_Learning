# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:34:07 2017

@author: dylan
"""

# setting seed
import os

os.environ['PYTHONHASHSEED'] = '0'
import numpy as np

SEED = 5  # 15,485,863
np.random.seed(SEED)
import random

random.seed(SEED)
RANDOM_STATE = np.random.RandomState(seed=SEED)
import pandas as pd

"""Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        # vision is given as a tensor with size of (64*64, 3) = (4096, 3) <-- rgb
        # and values are in [0, 255]
        
        name          dimension range  unit                             description
        focus:          (1,5)   [0,1] (0.05 m)     Vector of 5 range finder sensors: each sensor returns the distance
                                                    between the track edge and the car within a range of
                                                    200 meters
        
        speedX         scaler   (−∞,+∞) (km/h)     Speed of the car along the longitudinal axis of the car.
        
        speedY         scaler   (−∞,+∞) (km/h)     Speed of the car along the transverse axis of the car.
        
        speedZ         scaler   (−∞,+∞) (km/h)     Speed of the car along the Z axis of the car.
        
        opponents:      (1,36)  [0,1] (0.05 m)     Vector of 36 opponent sensors: each sensor covers a span
                                                    of 10 degrees within a range of 200 meters and returns the
                                                    distance of the closest opponent in the covered area
        rpm:            scalar    [0,+∞) (rpm)  Number of rotation per minute of the car engine.
        
        wheelSpinVel:   (1,4)  [0,+∞] (rad/s)   Vector of 4 sensors representing the rotation speed of
                                                   wheels.
        track:          (1,19) [0,1] (0.05 m)      Vector of 19 range finder sensors: each sensors returns the
                                                    distance between the track edge and the car within a range
                                                    of 200 meters.
        """
observations = ['focus', 'speedX', 'speedY', 'speedZ', 'opponents', 'rpm', 'track', 'wheelSpinVel']


# Initialize constants
ACTION_TYPE = 'Continuous'
ACTION_DIM = 2
ORIENTATION_DIM = 4
INPUT_DIM = [[5], [3], [1], [4]]


POLICY = 'softmax'
POLICY_LEARNING_ALGO = 'Q-learning'
GAMMA = 0.75
TAU = 1e-3
LR = 1e-3
BETA = 1e-3
TARGET_MODEL = 1
VANILLA = 0

THETAS = [0.6, 1.0]
MEWS = [0.0, 0.5]
SIGMAS = [0.3, 0.1]

TRAIN = 0
BATCH_SIZE = 5
ITERATIONS = 40000

NUM_STEPS = 50
NUM_EPISODES = 200

LOAD_MODEL = 0
DEBUG = 1
DISPLAY = 1


import matplotlib
if not DISPLAY:
    matplotlib.use('Agg')


# Preprocessing functions

def action_encoder(action_index):
    encoded_action = np.zeros(ACTION_DIM)
    encoded_action[action_index] = 1
    return encoded_action

def ou(action, theta, mew, sigma):
    """
    Ornstein-Uhlenbeck process
    :param action: float
    :param theta: float (+ve)
    :param mew: float
    :param sigma: float
    :return:
    """
    return theta*(action-mew) + RANDOM_STATE.randn()*sigma

def observation_formatter(observation, action=None):
    focus = np.reshape(observation[0], (-1, INPUT_DIM[0]))
    speedX = observation[1]
    speedY = observation[2]
    speedZ = observation[3]
    speed = np.reshape([speedX, speedY, speedZ], (-1, INPUT_DIM[1]))
    rpm = np.reshape(observation[5], (-1, INPUT_DIM[2]))
    wheelVel = np.reshape(observation[7], (-1, INPUT_DIM[3]))
    if action is None:
        return [focus, speed, rpm, wheelVel]
    else:
        action = np.reshape(action, (-1, ACTION_DIM))
        return [focus, speed, rpm, wheelVel, action]


def batch_observation_formatter(observations, actions=None):
    tmp = actions
    actions = np.repeat(None, len(observations)) if actions is None else actions
    formatted_states = []
    for i, state in enumerate(observations):
        formatted_states.append(observation_formatter(observations, actions[i]))
    formatted_states = np.matrix(formatted_states)
    if tmp is None:
        return [np.array(formatted_states[:, 1]), np.array(formatted_states[:, 2]), np.array(formatted_states[:, 3]), np.array(formatted_states[:, 4])]
    else:
        return [np.array(formatted_states[:, 1]), np.array(formatted_states[:, 2]), np.array(formatted_states[:, 3]), np.array(formatted_states[:, 4]), np.array(formatted_states[:, 5])]



# Preprocessing
# data = pd.read_csv('log_data_episode_5000.csv')
# # dic = {100: 1, -10: -0.2, -1: -0.1 }
# # data['Reward'] = data['Reward'].apply(lambda x: dic[x])
# data['Previous State'] = data['Previous State'].apply(clean_state).values
# data['Current State'] = data['Current State'].apply(clean_state).values

# Neural Network imports
import tensorflow as tf

tf.set_random_seed(SEED)
from keras.models import Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import concatenate
from keras.layers import add
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras import backend as K


# Neural Network functions
def entropy(policy):
    return -tf.reduce_sum(policy * tf.nn.log_softmax(policy))


def basic_actor_model():
    #track = Input(shape=INPUT_DIM[3]) Use either track or focus
    #opponents = Input(shape=[INPUT_DIM[2]]) Most likely won't use as agent is racing by itself

    focus = Input(shape=INPUT_DIM[0])
    speed = Input(shape=INPUT_DIM[1]) # vector of speedX, speedY and speedZ
    rpm = Input(shape=INPUT_DIM[2])
    wheelSpinVel = Input(shape=INPUT_DIM[3])

    speedh1 = Dense(32, activation='linear')(speed)
    wheelSpinVelh1 = Dense(32, activation='linear')(wheelSpinVel)
    combinedSpeed = concatenate([speedh1, wheelSpinVelh1, rpm])

    focush1 = Dense(128, activation='tanh')(focus)
    combinedSpeedh2 = Dense(128, activation='tanh')(combinedSpeed)
    combined_layer = concatenate([focush1, combinedSpeedh2])

    h3 = Dense(512, activation='relu')(combined_layer)

    steering = Dense(1, activation='tanh')(h3) #consider adding acceleration as an input to steering
    acceleration = Dense(1, activation='sigmoid')(h3)

    model = Model(inputs=[focus, speed, rpm, wheelSpinVel], outputs=[steering, acceleration])
    model.compile(loss='mse', optimizer=Adam(lr=1e-3))
    return model

def basic_critic_model():
    #track = Input(shape=INPUT_DIM[3]) Use either track or focus
    #opponents = Input(shape=[INPUT_DIM[2]]) Most likely won't use as agent is racing by itself

    actions = Input(shape=[ACTION_DIM])
    focus = Input(shape=INPUT_DIM[0])
    speed = Input(shape=INPUT_DIM[1]) # vector of speedX, speedY and speedZ
    rpm = Input(shape=INPUT_DIM[2])
    wheelSpinVel = Input(shape=INPUT_DIM[3])

    speedh1 = Dense(32, activation='linear')(speed)
    wheelSpinVelh1 = Dense(32, activation='linear')(wheelSpinVel)
    combinedSpeed = concatenate([speedh1, wheelSpinVelh1, rpm])

    focush1 = Dense(128, activation='tanh')(focus)
    combinedSpeedh2 = Dense(128, activation='tanh')(combinedSpeed)
    combined_layer = concatenate([focush1, combinedSpeedh2, actions])

    h3 = Dense(512, activation='relu')(combined_layer)

    Q = Dense(2, activation='linear')(h3)

    model = Model(inputs=[focus, speed, rpm, wheelSpinVel, actions], outputs=[Q])
    model.compile(loss='mse', optimizer=Adam(lr=1e-3))
    return model


def get_actor_update_operation(actor_model):
    policy = actor_model.outputs
    weights = actor_model.trainable_weights

    critic_gradients = tf.placeholder('float', shape=[None, ACTION_DIM])
    print(critic_gradients)
    gradients = tf.gradients(ys=policy, xs=weights, grad_ys=-critic_gradients)
    grads = zip(gradients, weights)
    operation = tf.train.AdamOptimizer(LR).apply_gradients(grads)

    return operation, critic_gradients


def get_gradient_operation(critic_model):
    Q_function = critic_model.outputs
    actions = critic_model.inputs[-1]
    print(actions)
    action_gradient_op = tf.gradients(Q_function, actions)
    return action_gradient_op


def get_critic_gradients(sess, gradient_op, critic_model, observations, actions):
    d = dict(zip(critic_model.inputs, batch_observation_formatter(observations, actions)))
    return sess.run([gradient_op], feed_dict=d)


def update_target_models(models):
    actor_model, critic_model, actor_target_model, critic_target_model = models[0], models[1], models[2], models[3]

    # Training actor target model
    weights = actor_model.get_weights()
    target_weights = actor_target_model.get_weights()
    for i in range(len(weights)):
        target_weights[i] = TAU * weights[i] + (1 - TAU) * target_weights[i]
    actor_target_model.set_weights(target_weights)

    # Training critic target model
    weights = critic_model.get_weights()
    target_weights = critic_target_model.get_weights()
    for i in range(len(weights)):
        target_weights[i] = TAU * weights[i] + (1 - TAU) * target_weights[i]
    critic_target_model.set_weights(target_weights)

def act(actor_model, observation):
    policy = actor_model.predict(observation)
    return ou(policy, THETAS, MEWS, SIGMAS)


def update_actor_critic_model(sess, models, episodes, tf_holders, iterations, batch_size):
    actor_model = models[0]
    critic_model = models[1]
    for iteration in range(iterations):
        if DEBUG:
            print('Iteration: {}'.format(iteration))
        else:
            print('\r', 'Iteration: {}'.format(iteration), end="")
        targets = []
        previous_observations = []
        actions = []
        deltas = []
        for _, frame in episodes.sample(batch_size, random_state=RANDOM_STATE).iterrows():
            previous_observation, actions, reward, observation, done = frame

            state = observation_formatter(previous_observation, actions)
            if TARGET_MODEL:
                q = models[3].predict(state, batch_size=1).flatten()
            else:
                q = critic_model.predict(state, batch_size=1).flatten()

            next_state = observation_formatter(observation)
            if TARGET_MODEL:
                next_policy = models[2].predict(next_state, batch_size=1).flatten()
            else:
                next_policy = actor_model.predict(next_state, batch_size=1).flatten()

            if done:
                target = reward
            else:
                if POLICY_LEARNING_ALGO == 'OU':
                    next_action = ou(next_policy, THETAS, MEWS, SIGMAS)

                next_action = np.reshape(next_action, (1, ACTION_DIM))
                if TARGET_MODEL:
                    future_q = models[3].predict(observation_formatter(observation, next_action), batch_size=1).flatten()
                else:
                    future_q = critic_model.predict(observation_formatter(observation, next_action), batch_size=1).flatten()
                target = reward + GAMMA * future_q

            actions.append(action)
            deltas.append(target - q)
            targets.append(target)
            previous_observations.append(previous_observation)

        critic_model.train_on_batch(batch_observation_formatter(previous_observations, actions), np.array(targets))
        gradients = get_critic_gradients(sess, tf_holders[2], critic_model, previous_observations, np.array(actions).squeeze(axis=1))
        print(gradients)
        gradients = np.squeeze(gradients)
        gradients = np.reshape(gradients, (-1, ACTION_DIM))

        d = dict(zip(actor_model.inputs, batch_observation_formatter(observations)))
        d[tf_holders[0]] = gradients
        sess.run([tf_holders[1]], feed_dict=d)

        if TARGET_MODEL:
            update_target_models(models)

        if DEBUG:
            print([observation, action])
            print(critic_model.predict(observation_formatter(observation, action), batch_size=1).flatten())
            print(actor_model.predict(observation_formatter(observation), batch_size=1).flatten())


K.set_learning_phase(1)  # set learning phase
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf)
K.set_session(sess)

from keras.models import load_model
#
# if LOAD_MODEL:
#     target_actor_model = load_model('actor_model_{}.h5'.format(ITERATIONS))
#     target_critic_model = load_model('critic_model_{}.h5'.format(ITERATIONS))
#     actor_model = load_model('actor_model_{}.h5'.format(ITERATIONS))
#     critic_model = load_model('critic_model_{}.h5'.format(ITERATIONS))
#     update_op, action_gradient_holder = get_actor_update_operation(actor_model)
#     gradient_op = get_gradient_operation(critic_model)
# else:

import sys
sys.path.append(os.path.abspath('../../../gym_torcs'))
from gym_torcs import TorcsEnv

#### Generate a Torcs environment
# enable vision input, the action is steering only (1 dim continuous action)
env = TorcsEnv(vision=True, throttle=False)

# without vision input, the action is steering and throttle (2 dim continuous action)
# env = TorcsEnv(vision=False, throttle=True)

# ob = env.reset()  # without torcs relaunch

# Generate an agent
actor_model = basic_actor_model()
critic_model = basic_critic_model()
if TARGET_MODEL:
    target_actor_model = basic_actor_model()
    target_critic_model = basic_critic_model()
else:
    target_actor_model = None
    target_critic_model = None


update_op, action_gradient_holder = get_actor_update_operation(actor_model)
gradient_op = get_gradient_operation(critic_model)
sess.run(tf.global_variables_initializer())
buffer = pd.DataFrame(columns=['previous observation', 'action', 'reward', 'observation'])

for episode in range(5):
    ob = env.reset(relaunch=True)  # with torcs relaunch (avoid memory leak bug in torcs)
    for move in range(10000):
        if TARGET_MODEL:
            action = act(target_actor_model, ob)
        else:
            action = act(actor_model, ob)
        new_ob, reward, done, _ = env.step(action)
        buffer.iloc[len(buffer), :] = [ob, action, reward, new_ob, done]
        ob = new_ob
        update_actor_critic_model(sess, [actor_model, critic_model, target_actor_model, target_critic_model], buffer,
                                  [action_gradient_holder, update_op, gradient_op], ITERATIONS, BATCH_SIZE)
        if done:
            break
# shut down torcs
env.end()


# from MarkovDecisionProcess import MDP
#
# mdp = MDP(LENGTH_OF_MAZE ** 2, ACTION_DIM, state_formatter=to_vanilla_state_formatter if VANILLA else state_formatter,
#           method='actor-critic', policy_type=POLICY, actor_model=target_actor_model, critic_model=target_critic_model,
#           target_models=[actor_model, critic_model], sess=sess, random_state=RANDOM_STATE)
# mdp.toolkit.set_formatters(state_formatter=state_formatter, batch_state_formatter=batch_state_formatter)
# mdp.toolkit.set_actor_update_op(actor_update_op=update_op, critic_gradient_holder=action_gradient_holder)
# mdp.toolkit.set_critic_gradient_operation(critic_gradient_op=gradient_op)
#
# from random_maze_environment import random_maze
#
# env = random_maze(LENGTH_OF_MAZE, NUM_COLOURS, randomize_maze=1, randomize_state=0, random_state=RANDOM_STATE)
# mdp.evaluate_model_in_environment(env, NUM_EPISODES, NUM_STEPS, show_env=DISPLAY, train=TRAIN)
#
# # Saving models
# mdp.actor_model.save('actor_model_{}.h5'.format(ITERATIONS))
# mdp.critic_model.save('critic_model_{}.h5'.format(ITERATIONS))

# MDP.evaluate_maze_model(model=actor_model, policy_type=POLICY, method='policy-network', complex_input=0,state_formatter=vanilla_state_formatter )

