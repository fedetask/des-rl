import numpy as np


def split_replay_batch(samples):
    """Takes a list of transitions (s, a, r, s') and splits them into arrays.

    Args:
        samples (list): List of tuples (s, a, r, s'), where:
            - s: numpy array, all s must have the same shape.
            - a: numpy array, all a must have the same shape.
            - r: float.
            - s': numpy array, all non None s' must have the same shape.

    Returns:
        A tuple (states, actions, rewards, next_states, next_states_idx) where each element
        is a numpy array with correspondent data in the rows. Pay attention that the rewards
        array is a column vector.

        Since the next state can be None when episode ends, the number of elements in next_states
        can be < len(samples). For this reason, next_states_idx is a numpy array that contains
        the actual indices of elements in next_states.
    """
    states = np.array([transition[0] for transition in samples])
    actions = np.array([transition[1] for transition in samples])
    rewards = np.array([transition[2] for transition in samples])
    if len(rewards.shape) == 1:
        rewards = rewards.reshape((rewards.shape[0], 1))

    next_states = []
    next_states_idx = []
    for i, transition in enumerate(samples):
        if transition[3] is not None:
            next_states.append(transition[3])
            next_states_idx.append(i)
    next_states = np.array(next_states)
    next_states_idx = np.array(next_states_idx)
    return states, actions, rewards, next_states, next_states_idx
