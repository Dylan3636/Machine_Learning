import numpy as np
from time import sleep
def evaluate_agent_in_environment(agent, env, num_episodes, num_timesteps, show_env=0, train=0, show_graph=0, reward_formatter=lambda x:x[1], delay=0, is_converged=lambda x: False,
                                  ax_env=None, ax_graph=None,
                                  verbose=1):
    """
    :param env: Environment the model is being evaluated on
    :param num_episodes: (int) Number of episodes to run for
    :param num_timesteps: (int) Nunmber of steps in each episode
    :param show_env(Boolean or list of integers): Flag indicating when and whether to render environment
    :param show_graph(Boolean or list of integers): Flag indicatin whether or when to plot graph of average returns
    :param reward_formatter function to alter reward
    :param ax_env axis to display environment on
    :param ax_graph axis to display graph on
    :param verbose: (int) Flag indicating what to print
    :return:
    """

    episodes = range(num_episodes)
    timesteps = range(num_timesteps)

    if type(show_env) is int:
        visualize = [[], list(episodes)][show_env]
    else:
        visualize = list(show_env)
    visualize_env = lambda x: x in visualize

    if type(show_graph) is int:
        episode_list_graph = [[], list(episodes)][show_graph]
    else:
        episode_list_graph = list(show_graph)
    visualize_graph = lambda x: x in episode_list_graph

    current_return = 0
    timesteps_list = []
    returns = []
    for episode in episodes:
        prev_observation = env.reset()

        if visualize_env(episode):
            env.render()
            sleep(delay)

        for t in timesteps:
            action = agent.make_decision(prev_observation)
            observation, reward, done, _ = env.step(action)
            reward = reward_formatter([observation, reward, done, t])
            if train:
                agent.update(state=prev_observation, action=action, reward=reward, next_state=observation, done=done, log=False, replay=False)
            current_return += reward
            prev_observation = observation.copy()

            if visualize_env(episode):
                env.render()
                sleep(delay)
            if visualize_graph(episode):
                agent.draw_graph(returns, ax=ax_graph)

            if done:
                returns.append(current_return)
                timesteps_list.append(t)
                ave_return = np.mean(returns[-(np.min([100, len(returns) - 1]))::])
                if verbose == 1:
                    print(
                        'Episode {} finished after {} timesteps. Total reward {}. Average return over past 100 episodes {}'.format(
                            episode, t, current_return, ave_return))
                current_return = 0
                break
        if is_converged(ave_return):
            break
    if train:
        timesteps_list=timesteps_list[-100::]
        returns = returns[-100::]
    stats = [int(np.mean(timesteps_list)), int(np.std(timesteps_list)), float(np.mean(returns)), float(np.std(returns))]
    print('Average timesteps per episode: {:d} +- {:d}\nAverage return: {:.3f} +/- {:.5f}'.format(stats[0], stats[1],
                                                                                                  stats[2], stats[3]))
    return stats
