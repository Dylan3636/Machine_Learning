import gym
import numpy as np
from agents.MarkovDecisionProcess import MDP
from environments.tools import evaluate_agent_in_environment

env_name ='CartPole-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
actions = range(env.action_space.n)
initial_weights = np.random.rand(15, 2)

def sig(a):
    return 1/(1 + np.exp(-a))

def rbf(x,c,h):
    return np.exp(-((x-c)/h)**2)


def rbf_formatter(x, cs, hs):
    if len(hs)==1:
        hs = np.repeat(hs, len(cs))
    return [rbf(x, c, h) for (c, h) in zip(cs, hs)]

def state_formatter(observations):
    x, x_dot, theta, theta_dot = observations
    x_rbf = rbf_formatter(x, cs=[-3, -2, -1, 0, 1, 2, 3], hs=[2])
    theta_rbf = rbf_formatter(theta, cs=[-0.2, -0.1, 0, 0.1, 0.2], hs=[0.2])
    return np.concatenate([[1], x_rbf, [x_dot/10], theta_rbf, [theta_dot/10]])


def reward_formatter(observation):
    _, reward, done, t = observation
    if done and t != 199:
        reward = 0
    else:
        reward /= 1
    return reward

num_episodes = 5000
num_timesteps = 200
episodes = range(num_episodes)
timesteps = range(500)
visualize = np.zeros(num_episodes)
for i in range(0, num_episodes, 50):
    visualize[i] = 1

mdp = MDP(num_actions=env.action_space.n, state_formatter=state_formatter, init_weights=initial_weights,lr=5e-3, alpha=0, method='q-linear', epsilon=1.0, gamma=0.99, epsilon_decay=0.9997, minimum_epsilon=0)
evaluate_agent_in_environment(mdp, env, num_episodes, num_timesteps=400, show_env=list(visualize), train=1, reward_formatter=reward_formatter, delay=0.05)