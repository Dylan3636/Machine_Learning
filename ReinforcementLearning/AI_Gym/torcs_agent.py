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
INPUT_DIM = [[21], [3], [1], [4]]


POLICY = 'OU'
GAMMA = 0.8
TAU = 1e-3
LR = 1e-4
EPSILON = 1.0
EPSILON_DECAY = 0.9998
MINIMUM_EPSILON = 0.2
TARGET_MODEL = 1
VANILLA = 0

THETAS = np.array([0.6, 1.0])
MEWS = np.array([0.0, 0.5])
SIGMAS = np.array([0.3, 0.1])

TRAIN = 0
BATCH_SIZE =20 
MINI_BATCH_SIZE=5
ITERATIONS = 1

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

def ou(action, theta, mu, sigma):
    """
    Ornstein-Uhlenbeck process
    :param action: float
    :param theta: float (+ve)
    :param mew: float
    :param sigma: float
    :return:
    """
    action = np.array(action)
    return theta*(mu-action) + RANDOM_STATE.randn(ACTION_DIM)*sigma

def observation_formatter(observation, action=None):
    trackPos = observation.trackPos
    angle = observation.angle
    comb = [trackPos, angle]
    focus = np.reshape(np.append(observation.track.flatten(), comb), (1, INPUT_DIM[0][0]))
    speedX = observation.speedX
    speedY = observation.speedY
    speedZ = observation.speedZ
    speed = np.reshape([speedX, speedY, speedZ], (1, INPUT_DIM[1][0]))
    rpm = np.reshape(observation.rpm, (1, INPUT_DIM[2][0]))
    wheelVel = np.reshape(observation.wheelSpinVel/100, (1, INPUT_DIM[3][0]))
    if action is None:
        return [focus, speed, rpm, wheelVel]
    else:
        action = np.reshape(action, (1, ACTION_DIM))
        return [focus, speed, rpm, wheelVel, action]


def batch_observation_formatter(observations, actions=None):
    tmp = actions
    actions = np.repeat(None, len(observations)) if actions is None else actions
    focuses = []
    speeds = []
    rpms = []
    wheelVels = []
    for i, observation in enumerate(observations):
        focus, speed, rpm, wheelVel = observation_formatter(observation)
        focuses.append(focus)
        speeds.append(speed)
        rpms.append(rpm)
        wheelVels.append(wheelVel)
    if tmp is None:
        return [np.reshape(focuses, (-1, INPUT_DIM[0][0])), np.reshape(speeds, (-1, INPUT_DIM[1][0])), np.reshape(rpms, (-1, INPUT_DIM[2][0])), np.reshape(wheelVels, (-1, INPUT_DIM[3][0]))]

    else:
        return [np.reshape(focuses, (-1, INPUT_DIM[0][0])), np.reshape(speeds, (-1, INPUT_DIM[1][0])), np.reshape(rpms, (-1, INPUT_DIM[2][0])), np.reshape(wheelVels, (-1, INPUT_DIM[3][0])), np.reshape(actions, (-1, ACTION_DIM))]



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
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten
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

    speed_rpm = concatenate([speed, rpm])
    speedh1 = Dense(150, activation='linear', init='glorot_normal')(speed_rpm)
    wheelSpinVelh1 = Dense(150, activation='linear', init='glorot_normal')(wheelSpinVel)
    combinedSpeed = add([speedh1, wheelSpinVelh1])

    focush1 = Dense(150, activation='linear')(focus)
    combined_layer = concatenate([focush1, combinedSpeed])

    h2 = BatchNormalization()(combined_layer)

    h3 = Dense(600, activation='relu')(h2)

    steering = Dense(1, activation='tanh', init='glorot_normal')(h3) #consider adding acceleration as an input to steering
    acceleration = Dense(1, activation='sigmoid', init='glorot_normal')(h3)

    output = concatenate([steering, acceleration])

    model = Model(inputs=[focus, speed, rpm, wheelSpinVel], outputs=[output])
    model.compile(loss='mse', optimizer=Adam(lr=1e-4))
    print(model.summary())
    return model

def basic_critic_model():
    #track = Input(shape=INPUT_DIM[3]) Use either track or focus
    #opponents = Input(shape=[INPUT_DIM[2]]) Most likely won't use as agent is racing by itself

    actions = Input(shape=[ACTION_DIM])
    focus = Input(shape=INPUT_DIM[0])
    speed = Input(shape=INPUT_DIM[1]) # vector of speedX, speedY and speedZ
    rpm = Input(shape=INPUT_DIM[2])
    wheelSpinVel = Input(shape=INPUT_DIM[3])

    focus = Input(shape=INPUT_DIM[0])
    speed = Input(shape=INPUT_DIM[1]) # vector of speedX, speedY and speedZ
    rpm = Input(shape=INPUT_DIM[2])
    wheelSpinVel = Input(shape=INPUT_DIM[3])

    speed_rpm = concatenate([speed, rpm])
    speedh1 = Dense(150, activation='linear', init='glorot_normal')(speed_rpm)
    wheelSpinVelh1 = Dense(150, activation='linear', init='glorot_normal')(wheelSpinVel)
    combinedSpeed = add([speedh1, wheelSpinVelh1])

    focush1 = Dense(150, activation='linear')(focus)
    combined_layer = concatenate([focush1, combinedSpeed])

    h2 = BatchNormalization()(combined_layer)
    h3 = Dense(600, activation='relu')(h2)

    action_h = BatchNormalization()(Dense(600, activation='linear', init='glorot_normal')(actions))
    combined = add([h3, action_h])

    final = Dense(600, activation='relu')(BatchNormalization()(combined))
    Q = Dense(2, activation='linear', init='glorot_normal')(BatchNormalization()(final))

    model = Model(inputs=[focus, speed, rpm, wheelSpinVel, actions], outputs=[Q])
    model.compile(loss='mse', optimizer=Adam(lr=1e-3))
    print(model.summary())
    return model


def get_actor_update_operation(actor_model):
    policy = actor_model.outputs
    weights = actor_model.trainable_weights
    critic_gradients = tf.placeholder('float',[None,ACTION_DIM])
    gradients = tf.gradients(ys=policy, xs=weights, grad_ys=-critic_gradients)
    grads = zip(gradients, weights)
    operation = tf.train.AdamOptimizer(LR).apply_gradients(grads)
    return operation, critic_gradients


def get_gradient_operation(critic_model):
    Q_function = critic_model.outputs
    actions = critic_model.inputs[-1]
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
    policy = actor_model.predict(observation).flatten()
    action= np.array(policy) + EPSILON*ou(policy, THETAS, MEWS, SIGMAS)
    action[0] = max(min(action[0], 1), -1)
    action[1] = max(min(action[1], 1), 0)
    print('policy:', policy)
    print('action:', action)
    return action


def update_actor_critic_model(sess, models, episodes, tf_holders, iterations, batch_size):
    actor_model = models[0]
    critic_model = models[1]
    for iteration in range(iterations):
        targets = []
        previous_observations = []
        observations = []
        actions = []
        next_actions = []
        deltas = []
        rewards=[]
        id_mask = []
        for _, frame in episodes.sample(min(batch_size,len(episodes)), random_state=RANDOM_STATE).iterrows():
            previous_observation, action, reward, observation, done = frame

            state = observation_formatter(previous_observation, action)
            if TARGET_MODEL:
                q = models[3].predict(state, batch_size=1).flatten()
            else:
                q = critic_model.predict(state, batch_size=1).flatten()

            next_state = observation_formatter(observation)

            if done:
                id_mask.append(np.zeros(ACTION_DIM))
                target = reward*np.ones(ACTION_DIM)
                next_action=[0,0]
            else:
                id_mask.append(1)
                if POLICY == 'OU':
                   if TARGET_MODEL:
                      next_action = act(models[2], next_state)
                   else:
                      next_action = act(actor_model, next_state)
            

                next_action = np.reshape(next_action, (1, ACTION_DIM))
                if TARGET_MODEL:
                    future_q = models[3].predict(observation_formatter(observation, next_action), batch_size=1).flatten()
                else:
                    future_q = critic_model.predict(observation_formatter(observation, next_action), batch_size=1).flatten()
                target = reward + GAMMA * future_q

            rewards.append(reward)
            actions.append(action)
            next_actions.append(next_action)
            deltas.append(target - q)
            targets.append(target)
            observations.append(observation)
            previous_observations.append(previous_observation)
        #print('targets: ', targets)
        critic_model.train_on_batch(batch_observation_formatter(previous_observations, actions), np.array(targets))
        gradients = np.squeeze(get_critic_gradients(sess, tf_holders[2], critic_model, previous_observations, np.array(actions)))
        #future_gradients = np.squeeze(get_critic_gradients(sess, tf_holders[2], critic_model, observations, np.array(actions)))
        #gradients = np.reshape(np.array(rewards) + np.array(id_mask)*GAMMA*future_gradients-gradients, (1, ACTION_DIM))
        #print('gradients: ', gradients)

        d1 = dict(zip(actor_model.inputs, batch_observation_formatter(previous_observations)))
        d1[tf_holders[0]] = np.reshape(gradients, [-1, ACTION_DIM])
        d = dict( d1)
        sess.run([tf_holders[1]], feed_dict=d)

        if TARGET_MODEL:
            update_target_models(models)

        if DEBUG:
            pass
            #print([observation, action])
            #print(critic_model.predict(observation_formatter(observation, action), batch_size=1))
            #print(actor_model.predict(observation_formatter(observation), batch_size=1))


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
#env = TorcsEnv(vision=False, throttle=False)

# without vision input, the action is steering and throttle (2 dim continuous action)
env = TorcsEnv(vision=False, throttle=True)

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
buffer = pd.DataFrame(columns=['previous observation', 'action', 'reward', 'observation', 'done'])

def safe_norm(x):
    xmax = np.max(x)
    return np.linalg.norm(x / xmax) * xmax

for episode in range(4000):
    print('Episode: ', episode)
    if episode %1 ==0:
        ob = env.reset(relaunch=True)  # with torcs relaunch (avoid memory leak bug in torcs)
    else:
        ob = env.reset()
    for move in range(10000):
        if TARGET_MODEL:
            action = act(target_actor_model, observation_formatter(ob))
        else:
            action = act(actor_model, observation_formatter(ob))
        action = action.flatten()
        new_ob, reward, done, _ = env.step(action)
        reward = reward/400
        print('\nq-value: ', target_critic_model.predict(observation_formatter(ob, action)))
        print('reward: ', reward, '\n')
        if np.isnan(reward):
            break
        buffer.loc[len(buffer), :] = [ob, action, reward, new_ob, done]
        update_actor_critic_model(sess, [actor_model, critic_model, target_actor_model, target_critic_model], buffer,
                                  [action_gradient_holder, update_op, gradient_op], ITERATIONS, BATCH_SIZE)
        ob = new_ob
        EPSILON = max(EPSILON*EPSILON_DECAY, MINIMUM_EPSILON)
        #print('\nepsilon: ', EPSILON, '\n')
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

