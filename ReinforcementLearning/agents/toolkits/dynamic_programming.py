import numpy as np

def value_iteration(transition_probabilities, immediate_rewards, actions, gamma=0.9, theta=0.1,
                    terminal_states=[], in_place=True, initial_policy=None):
    """

    :param transition_probabilities:
    :param immediate_rewards:
    :param actions:
    :param gamma:
    :param theta:
    :param terminal_states:
    :param in_place:
    :param initial_policy:
    :return:

    >>> import numpy as np
    >>> policy = np.array([[0.5, 0.5], [0.5, 0.5]])
    >>> transition_probabilities = np.array([np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0, 1], [0, 1]])])
    >>> transition_model = lambda a, st: transition_probabilities[a] if st else [state[a] for state in transition_probabilities]
    >>> immediate_rewards = [-10, 1]
    >>> print('Values:{val[0]}, Policy:{val[1]}'.format(val = MDP.value_iteration(transition_model, immediate_rewards, 2, theta=1e-10)))
        Values:[-5.5112429  4.0951   ], Policy:[array([ 1.,  0.]), array([ 1.,  0.])]

    """
    num_states = np.size(transition_probabilities(0,False), 0)
    repeat = 3 - np.ndim(immediate_rewards)

    if type(actions) is int:
        num = actions
        actions = []
        for _ in range(num_states):
            actions.append(range(0, num))

    values = np.random.rand(num_states)
    #values[-1] = immediate_rewards[-1]
    if repeat == 1:
        # R(s,a) --> R(s,a,s)
        sas = np.repeat([immediate_rewards], num_states, axis = 0)
        ir = lambda i : sas[i]
        repeat = 0
    elif repeat==2:
        sas = []
        for i in range(num_states):
                # R(s) --> R(s,a,s)
                tmp = np.repeat([immediate_rewards], np.size(actions[i],0), axis = 0)
                sas.append(tmp)
        ir = lambda i : sas[i]
        repeat = 0
    else:
        ir = lambda i: immediate_rewards[i]

    if in_place:
        new_values = np.array([num_states, 1])
    if initial_policy is None:
        policy = []
        for state_actions in actions:
            policy.append(np.eye(1, len(state_actions)))
    else:
        policy = initial_policy
    stored = False
    tp_ir = []
    while True:
        delta = -np.inf
        for i in range(0, num_states):
            if i in terminal_states:
                continue

            if not stored:
                tp_ir.append(np.sum(transition_probabilities(i, True) * ir(i), 1))

            tmp = tp_ir[i] + np.dot(transition_probabilities(i, True), ( gamma * values))
            new_v = np.max(tmp)
            policy[i] = np.zeros(len(actions[i]))
            policy[i][np.argmax(tmp)] = 1
            delta = max(delta, np.abs(values[i] - new_v))
            if not in_place:
                new_values[i] = new_v
            else:
                values[i] = new_v
        if not in_place:
            values = new_values
        if delta < theta:
            break
        stored = True
    return values, policy

def policy_evaluation(policy, transition_model, immediate_rewards, actions, gamma=0.9, theta=0.1, terminal_states=[],
                      in_place=True):

    num_states = np.size(transition_model(0,False), 0)
    repeat = 3 - np.ndim(immediate_rewards)

    if type(actions) is int:
        num = actions
        actions = []
        for _ in range(num_states):
            actions.append(list(range(0, num)))

    values = np.random.rand(num_states)
    #values[-1] = immediate_rewards[-1]
    if in_place:
        new_values = np.array([num_states, 1])
    while True:
        delta = -np.inf
        for i in range(0, num_states):
            if i in terminal_states:
                continue
            new_v = 0
            for j in actions[i]:

                if repeat == 1:
                    # R(s,a) --> R(s,a,s)
                    ir = np.repeat([immediate_rewards], num_states, axis =0)
                    repeat = 0
                elif repeat == 2:
                    # R(s) --> R(s,a,s)
                    ir = np.repeat([immediate_rewards], np.size(actions[i],0), axis = 0)
                    ir = np.repeat([ir], num_states, axis =0)

                transition_probabilities = transition_model(j, False)
                new_v += policy[i][j] * np.dot(transition_probabilities[i], (ir[i][j] + (gamma * values)))
            delta = max(delta, np.abs(values[i] - new_v))
            if not in_place:
                new_values[i] = new_v
            else:
                values[i] = new_v
        if not in_place:
            values = new_values
        # print values
        if delta < theta:
            break
    return values


def policy_iteration(initial_policy, transition_probabilities, immediate_rewards, actions, gamma=0.9,
                     theta=0.1, terminal_states=[], in_place=True):
    """

    :param initial_policy:
    :param transition_probabilities:
    :param immediate_rewards:
    :param actions:
    :param gamma:
    :param theta:
    :param terminal_states:
    :param in_place:
    :return: values, policy

    >>> import numpy as np
    >>> policy = np.array([[0.5, 0.5], [0.5, 0.5]])
    >>> transition_probabilities = np.array([np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0, 1], [0, 1]])])
    >>> transition_model = lambda a, st: transition_probabilities[a] if st else [state[a] for state in transition_probabilities]
    >>> immediate_rewards = [-10, 1]
    >>> print('Values:{val[0]}, Policy:{val[1]}'.format(val = MDP.policy_iteration(policy, transition_model, immediate_rewards, 2, theta=1e-10)))
        Values:[-5.52905259  4.0951    ], Policy:[[ 1.  0.] [ 1.  0.]]
    """
    num_states = np.size(transition_probabilities(0,False), 0)
    repeat = 3 - np.ndim(immediate_rewards)

    if type(actions) is int:
        num = actions
        actions = []
        for _ in range(num_states):
            actions.append(range(0, num))

    if repeat == 1:
        # R(s,a) --> R(s,a,s)
        sas = np.repeat([immediate_rewards], num_states, axis = 0)
        ir = lambda i : sas[i,:,:]
        repeat = 0
    elif repeat==2:
        sas = []
        for i in range(num_states):
                # R(s) --> R(s,a,s)
                tmp = np.repeat([immediate_rewards], np.size(actions[i], 0), axis = 0)
                sas.append(tmp)
        ir = lambda i : sas[i]
        repeat = 0
    else:
        ir = lambda i: immediate_rewards[i,:,:]


    if initial_policy is None:
        policy = []
        for state_actions in actions:
            policy.append(np.eye(1, len(state_actions)))
    else:
        policy = initial_policy
    stored = False
    tp_ir = []
    while True:
        policy_stable = True
        values = policy_evaluation(policy, transition_probabilities, immediate_rewards, actions, gamma,
                                       theta, terminal_states, in_place)  # Policy Evaluation
        for i in range(0, num_states):
            b = np.argmax(
                policy[i])  # policy should have an indicator vector for each state indicating the action to be taken
            policy[i] = np.zeros(len(actions[i]))

            if not stored:
                 tp_ir.append(np.sum(transition_probabilities(i, True) * ir(i), 1))
            tmp = tp_ir[i] + np.dot(transition_probabilities(i, True), ( gamma * values))

            ind = np.argmax(tmp)
            policy[i][ind] = 1
            if ind != b:
                policy_stable = False
            stored
        if policy_stable:
            break

    return values, policy
