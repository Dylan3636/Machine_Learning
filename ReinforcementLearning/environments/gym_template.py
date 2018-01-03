import gym

env_name ='CartPole-v0'
env = gym.make(env_name)
visualize = 1
episodes = range(50)
timesteps = range(100)
print('Action space: {}'.format(env.action_space))
print('Observation space: {}'.format(env.observation_space))

for episode in episodes:
    observation = env.reset() # initial previous_observation
    for t in timesteps:
        if visualize:
            env.render()
        action = env.action_space.sample() # pick action
        observation, reward, done, info = env.step(action) # take action and see results

        # Do something with previous_observation.
        if done:
            print('Episode {} finished after {} timesteps'.format(episode, t))
            break
