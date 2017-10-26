import numpy as np
import gym
from agents.MarkovDecisionProcess import MDP
from environments.tools import evaluate_agent_in_environment
env = gym.make('CarRacing-v0')
for i in range(1000):
    env.reset()
    for j in range(300):
        env.render()
        action = np.random.rand(3)
        obs,_,done,_=env.step(action)
        print(np.shape(obs))
        if done:
            break
# agent = MDP(num_actions= 0, policy_type='random')
# evaluate_agent_in_environment(agent, env, 1000, 200, 1)
