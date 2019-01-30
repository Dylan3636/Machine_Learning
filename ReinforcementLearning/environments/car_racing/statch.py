# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# img = cv2.imread('car_racing_img_30.png')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgs = [img, img_gray]
# for i in range(30, 50):
#     img = cv2.imread('car_racing_img_{}.png'.format(i))
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print(img_gray.shape)
#     cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Gray', 600, 600)
#     cv2.imshow('Gray', img_gray)
#     k = cv2.waitKey(5) & 0xFF
#     time.sleep(1)

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 23:34:07 2017

credit to yanpanlau for initial implementation. (https://github.com/yanpanlau/DDPG-Keras-Torcs)
@author: dylan
"""

# setting seed
import os

os.environ['PYTHONHASHSEED'] = '0'
import numpy as np

SEED = 5
np.random.seed(SEED)
import random

random.seed(SEED)
RANDOM_STATE = np.random.RandomState(seed=SEED)
import pandas as pd
import cv2


# Initialize constants
ACTION_TYPE = 'Continuous'
ACTION_DIM = 3
INPUT_DIM = [[60, 60, 1], [7]]

POLICY = 'OU'
GAMMA = 0.99
TAU = 1e-3
LRA = 1e-4
LRC = 1e-3
EPSILON = 1.0
EPSILON_DECAY = 0.999985
MINIMUM_EPSILON = 0.2
TARGET_MODEL = 1
VANILLA = 0

THETAS = np.array([0.6, 1.0, 1.0])
MEWS = np.array([0.0, 0.6, 0.1])
SIGMAS = np.array([0.3, 0.1, 0.05])

TRAIN = 0
BATCH_SIZE = 16
ITERATIONS = 200

NUM_STEPS = 50
NUM_EPISODES = 200

LOAD_MODEL = 0
DEBUG = 1
DISPLAY = 1

import matplotlib

if not DISPLAY:
    matplotlib.use('Agg')


# Preprocessing functions

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
    return theta * (mu - action) + RANDOM_STATE.randn(ACTION_DIM) * sigma


def rbf(x, c, h):
    return tf.exp(-tf.pow(((x-c)/h), 2))


def rbf_formatter(x, cs, hs):
    if np.shape(hs)[0] == 1:
        hs = np.repeat(hs, len(cs), axis=0)
    return tf.concat([(rbf(x, c, h)) for (c, h) in zip(cs, hs)], axis=1)

def rbf_lambda_func(x):
    steer_cs = np.arange(0, 1.05, 0.05)
    accel_cs = np.arange(0, 1.05, 0.05)
    brake_cs = np.arange(0, 1.05, 0.05)
    cs = np.concatenate([np.reshape(steer_cs, (-1, 1)), np.reshape(accel_cs, (-1, 1)), np.reshape(brake_cs, (-1, 1))],
                   axis=1)
    rbfs = rbf_formatter(x, cs, 0.1*np.ones([1, 3]))
    return rbfs

def state_rbf_func(x):
    cs = np.repeat(np.reshape(np.arange(0, 1.05, 0.05), (-1, 1)), 7, axis=1)
    rbfs = rbf_formatter(x, cs, 0.1*np.ones([1, 7]))
    return rbfs

def compute_steering_speed_gyro_abs(a):
    right_steering = a[6, 36:46].mean() / 255
    left_steering = a[6, 26:36].mean() / 255
    steering = (right_steering - left_steering + 1.0) / 2

    left_gyro = a[6, 46:60].mean() / 255
    right_gyro = a[6, 60:76].mean() / 255
    gyro = (right_gyro - left_gyro + 1.0) / 2

    speed = a[:, 0][:-2].mean() / 255
    abs1 = a[:, 6][:-2].mean() / 255
    abs2 = a[:, 8][:-2].mean() / 255
    abs3 = a[:, 10][:-2].mean() / 255
    abs4 = a[:, 12][:-2].mean() / 255
    return [steering, speed, gyro, abs1, abs2, abs3, abs4]


def observation_formatter(observation, action=None):
    if len(observation) == 96:
        state = np.reshape(compute_steering_speed_gyro_abs(observation), (-1, 7))
        img_gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)[0:84, :]
        ret, mask = cv2.threshold(img_gray, 150, 200, cv2.THRESH_BINARY_INV)
        img_gray = cv2.resize(cv2.bitwise_and(img_gray, img_gray, mask=mask), (60, 60))
        img_gray = np.reshape(img_gray, (-1, INPUT_DIM[0][0], INPUT_DIM[0][1], 1))
        img_gray = (img_gray+100)/355
    else:
        img_gray, state = observation
    # cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Gray', 600, 600)
    # cv2.imshow('Gray', img_gray[0])
    # k = cv2.waitKey(5) & 0xFF

    if action is None:
        return [img_gray, state]
    else:
        action[0] = (action[0] + 1.0)/2
        action = np.reshape(action, (1, ACTION_DIM))
        return [img_gray, state, action]


def batch_observation_formatter(observations, actions=None):
    imgs = []
    states = []
    for i, observation in enumerate(observations):
        img, state = observation_formatter(observation)
        imgs.append(img)
        states.append(state)
    if actions is None:
        return [np.reshape(imgs, (-1, INPUT_DIM[0][0], INPUT_DIM[0][1], 1)), np.reshape(states, (-1, INPUT_DIM[1][0])) ]
    else:
        return [np.reshape(imgs, (-1, INPUT_DIM[0][0], INPUT_DIM[0][1], 1)), np.reshape(states, (-1, INPUT_DIM[1][0])), np.reshape(actions, (-1, ACTION_DIM))]


# Neural Network imports
import tensorflow as tf

tf.set_random_seed(SEED)
from keras.models import Input
from keras.models import Model
from keras.layers import concatenate
from keras.layers import add
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.initializers import RandomNormal
from keras.regularizers import l1, l2
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

# Neural Network functions
def entropy(policy):
    return -tf.reduce_sum(policy * tf.nn.log_softmax(policy))


def basic_actor_model():
    img = Input(INPUT_DIM[0])
    conv1 = BatchNormalization()(Conv2D(32, [3, 3], activation='relu')(img))
    conv2 = BatchNormalization()(Conv2D(32, [3, 3],strides=2, activation='relu')(conv1))
    conv3 = BatchNormalization()(Conv2D(32, [3, 3],strides=2, activation='relu')(conv2))
    flattened = Flatten()(conv3)
    h1 = Dropout(0.4)(LeakyReLU()(Dense(200)(flattened)))
    state = Input(INPUT_DIM[1])
    state_lambda = Lambda(function=state_rbf_func, output_shape=[147])(state)
    state_h1 = Dropout(0.1)(Dense(200, kernel_regularizer=l1(0.01))(state_lambda))
    combined = add([h1, state_h1])
    h2 = Dropout(0.2)(LeakyReLU()(Dense(200)(combined)))
    steering = Dense(1, activation='tanh', kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4))(h2)
    acceleration = Dense(1, activation='sigmoid', kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4))(h2)
    breaking = Dense(1, activation='sigmoid', kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4))(h2)
    output = concatenate([steering, acceleration, breaking])

    model = Model(inputs=[img, state], outputs=[output])
    model.compile(loss='mse', optimizer=Adam(lr=LRA))
    print(model.summary())
    return model


def basic_critic_model():

    img = Input(shape=INPUT_DIM[0])
    actions = Input(shape=[ACTION_DIM])

    conv1 = BatchNormalization()(Conv2D(32, [3, 3], activation='relu')(img))
    conv2 = BatchNormalization()(Conv2D(32, [3, 3],strides=2, activation='relu')(conv1))
    conv3 = BatchNormalization()(Conv2D(32, [3, 3],strides=2, activation='relu')(conv2))
    flattened = Flatten()(conv3)
    h1 = (Dense(200, activation='relu')(flattened))
    state = Input(INPUT_DIM[1])
    state_lambda = Lambda(function=state_rbf_func)(state)
    state_h1 = Dense(200, kernel_regularizer=l1(0.01))(state_lambda)
    combined = add([h1, state_h1])
    actions_rbf = (Lambda(rbf_lambda_func, output_shape=[63])(actions))
    h_actions = Dropout(0.2)(Dense(200, activation='linear', kernel_regularizer=l1(0.01))(actions_rbf))
    h2 = Dropout(0.2)(Dense(200, activation='relu', kernel_regularizer=l2(0.01))(add([combined, h_actions])))
    Q = Dense(ACTION_DIM, activation='linear', kernel_initializer='glorot_normal')(h2)
    model = Model(inputs=[img, state, actions], outputs=[Q])
    model.compile(loss='mse', optimizer=Adam(lr=LRC))
    print(model.summary())
    return model


def get_actor_update_operation(actor_model):
    policy = actor_model.outputs
    weights = actor_model.trainable_weights
    critic_gradients = tf.placeholder('float', [None, ACTION_DIM])
    gradients = tf.gradients(ys=policy, xs=weights, grad_ys=-critic_gradients)
    grads = zip(gradients, weights)
    operation = tf.train.AdamOptimizer(LRA).apply_gradients(grads)
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
    if np.random.rand() < 0.7*EPSILON:
        policy[2] = min(policy[2], 2*RANDOM_STATE.randn() + 0.2)
    action = np.array(policy) + EPSILON * ou(policy, THETAS, MEWS, SIGMAS)
    action[0] = max(min(action[0], 1), -1)
    action[1] = max(min(action[1], 1), 0)
    action[2] = max(min(action[2], 1), 0)
    return action, policy

from time import time

def update_actor_critic_model(sess, models, episodes, tf_holders, iterations, batch_size):
    actor_model = models[0]
    critic_model = models[1]
    # for iteration in range(iterations):
    #     print('Training iteration: {}'.format(iteration))
    frames = episodes.sample(batch_size)
    #previous_observation, action, reward, observation, done = frame
    previous_observations = frames.iloc[:, 0]
    actions = frames.iloc[:, 1]
    rewards = frames.iloc[:, 2]
    observations = frames.iloc[:, 3]

    next_states = batch_observation_formatter(observations)
    mask = np.logical_not(frames.iloc[:, 4])
    mask = mask.reshape(batch_size, 1)
    rewards = rewards.reshape(batch_size, 1)
    next_actions = models[2].predict_on_batch(next_states)
    future_qs = models[3].predict_on_batch(batch_observation_formatter(observations, next_actions))

    targets = rewards + ([GAMMA] *mask*future_qs)
    critic_model.train_on_batch(batch_observation_formatter(previous_observations, [actions]), np.array(targets))
    actions_for_grad = models[2].predict_on_batch(batch_observation_formatter(previous_observations))

    gradients = np.squeeze(
        get_critic_gradients(sess, tf_holders[2], critic_model, previous_observations, np.array(actions_for_grad)))

    #print(gradients)

    d1 = dict(zip(actor_model.inputs, batch_observation_formatter(previous_observations)))
    d1[tf_holders[0]] = np.reshape(gradients, [-1, ACTION_DIM])
    d = dict(d1)
    sess.run([tf_holders[1]], feed_dict=d)

    if TARGET_MODEL:
        #t = time()
        update_target_models(models)
        #print('t5: ', t - time())

    if DEBUG:
        pass
        # print([observation, action])
        # print(critic_model.predict(observation_formatter(observation, action), batch_size=1))
        # print(actor_model.predict(observation_formatter(observation), batch_size=1))


K.set_learning_phase(1)  # set learning phase
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf)
K.set_session(sess)

# Generate an agents
actor_model = basic_actor_model()
critic_model = basic_critic_model()
sess.run(tf.global_variables_initializer())
if TARGET_MODEL:
    from keras.models import load_model
    actor_model.save('initial_actor.h5')
    critic_model.save('initial_critic.h5')
    target_actor_model = load_model('initial_actor.h5', custom_objects={'state_rbf_func':state_rbf_func, 'rbf_formatter': rbf_formatter, 'rbf':rbf}) #basic_actor_model()
    target_critic_model = load_model('initial_critic.h5', custom_objects={'state_rbf_func':state_rbf_func, 'rbf_lambda_func':rbf_lambda_func, 'rbf_formatter': rbf_formatter, 'rbf':rbf})
else:
    target_actor_model = None
    target_critic_model = None

update_op, action_gradient_holder = get_actor_update_operation(actor_model)
gradient_op = get_gradient_operation(critic_model)
# target_actor_model.set_weights(actor_model.trainable_weights)
# target_critic_model.set_weights(critic_model.trainable_weights)

buffer = pd.DataFrame(columns=['previous observation', 'action', 'reward', 'observation', 'done'])
import gym
env = gym.make('CarRacing-v0')
sigmoid = lambda a:1-1/(1+np.exp(-((a+0.9999999999999886)/ 2.4666161984293584e-14)))
for episode in range(40000):
    print('Episode: ', episode)
    ob = env.reset()
    count=0
    for move in range(10000):
        #ti = time()
        print('Move: {}'.format(move))
        #t=time()
        env.render()
        #print('render time: ', time()-t)
        action, policy = act(target_actor_model, observation_formatter(ob))
        if move < 15:
            if RANDOM_STATE.rand()< EPSILON:
                action = np.array([0.0, 0.5, 0])
                action = np.array(action) + EPSILON * ou(action, THETAS, MEWS, SIGMAS)
                action[0] = max(min(action[0], 1), -1)
                action[1] = max(min(action[1], 1), 0)
                action[2] = max(min(action[2], 1), 0)
        if DEBUG:
            print('policy:', policy)
            print('action:', action)
        action = action.flatten()
        new_ob, reward, done, _ = env.step(action)
        t=time()
        new_ob = observation_formatter(new_ob)
        reward = 10*reward if reward < 0 else reward*2.5
        reward = -1 if reward == -1000 else reward
        val = observation_formatter(ob, policy)
        if np.sum(val[0]) < 1020 and move > 20:
            count += 1
            if count > 8:
                reward = -1
                done = True
        else:
            count = 0

        if reward < 0 and reward != -1:
            reward = -sigmoid(reward)
        #reward += max(np.log(policy[1]), -10)
        EPSILON = max(EPSILON * EPSILON_DECAY, MINIMUM_EPSILON)
        #t = time()
        buffer = buffer.append(dict(zip(['previous observation', 'action', 'reward', 'observation', 'done'], [ob, action, reward, new_ob, done])), ignore_index=1)
        #print('append time: ', time()-t)

        if DEBUG:
            print('\nq-value(target): ', target_critic_model.predict(val))
            print('q-value: ', critic_model.predict(val))
            print('reward: ', reward, '\n')

        #print(time() - prev_time)
        ob = new_ob
        if done:
            break
        # pos_reward = buffer[(buffer.reward > 0)]
        # neg_reward_sample = pd.DataFrame(buffer[buffer.reward < 0].sample(int(2*len(pos_reward)),replace =False, random_state=RANDOM_STATE))
        # buffer = pd.concat([pos_reward, neg_reward_sample], axis=0, ignore_index=True)
        buffer = buffer.iloc[int(-.5e6)::, :]
        # target_weights = targt_actor_model.get_weights()
        # print(target_weights)
        batch_size = min(len(buffer), BATCH_SIZE)
        #iterations = np.min([ITERATIONS, int(len(buffer)/batch_size)])
        #t=time()
        update_actor_critic_model(sess, [actor_model, critic_model, target_actor_model, target_critic_model], buffer,
                                  [action_gradient_holder, update_op, gradient_op], 1, batch_size)
        #print('t: ', time()-t)
        #print('total: ', time()-ti)

        # target_weights = target_actor_model.get_weights()
        # print(target_weights)
    if episode in [5, 10, 20, 30]:
        buffer.to_csv('car_racing_data_{}.csv'.format(episode))
    print('Epsilon: ', EPSILON)
