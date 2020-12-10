import os
import numbers

import torch
import numpy as np
import itertools


class ParameterUpdater:
    """ParameterUpdater class that stores a parameter and handles its update according to the
    selected schedule.

    For 'const' schedule, the parameter is left to its start value.
    For 'lin' schedule the parameter is updated as param(t) = start - t * (start - end) / n_steps.
    For 'exp' schedule, the parameter is updated as param(t) = alpha * exp(-beta * t),
    where alpha and beta are set such that param(0) = start and param(n_steps) = end.
    """

    VALID_SCHEDULES = ['const', 'lin', 'exp']

    def __init__(self, start, end, n_steps, update_schedule='lin'):
        """Create the ParameterUpdater.

        Args:
            start (float): Initial value of the parameter.
            end (float): Final value of the parameter.
            n_steps (int): Total number of steps over which the parameter is updated.
            update_schedule (str): Either 'const', 'lin', or 'exp'.
        """
        assert update_schedule in ParameterUpdater.VALID_SCHEDULES,\
            'The given schedule is not understood. Use ' + str(ParameterUpdater.VALID_SCHEDULES)
        if update_schedule == 'const':
            assert start == end, 'Constant schedule requires start==end.'
        if update_schedule == 'exp' and (start < 0 or end < 0):
            raise NotImplementedError('Exponential decay for negative values not implemented yet.')
        self.start = start
        self.end = end
        self.n_steps = n_steps
        self.update_schedule = update_schedule
        self.cur_value = start
        self.step = 0
        self.decreasing = start > end

    def update(self):
        """Perform one update step of the parameter, i.e. updating it from param(t) to param(t+1).

        Every update() call outside the given steps range will set the value to the given end value.
        """
        if self.update_schedule == 'const':
            pass
        elif self.update_schedule == 'lin':
            self.cur_value = self.start - (self.step + 1) * (self.start - self.end) / self.n_steps
        elif self.update_schedule == 'exp':
            beta = (-1. / self.n_steps) * np.log(self.end / self.start)
            self.cur_value = self.start * np.exp(-beta * (self.step + 1))
        else:
            raise ValueError(
                'The given update schedule \'' + str(self.update_schedule) + '\' is not understood.'
            )
        if self.step >= self.n_steps:
            self.cur_value = self.end
        self.step += 1

    def get_value(self):
        """Returns: The current value of the parameter.
        """
        return self.cur_value


def clamp(tensor, min_value, max_value):
    """Clamp the tensor in the given range.

    Args:
        tensor (torch.Tensor): Tensor whose elements have to be clipped.
        min_value (Union[float, torch.Tensor, numpy.ndarray]): Low clipping value. If not a
            scalar, its shape must match tensor and the clipping is be performed element wise.
        max_value (Union[float, torch.Tensor]): High clipping value. If a Tensor is given,
            its shape must match tensor and the clipping is be performed element wise.

    Returns:
        A copy of the given tensor where each element is clipped in the given range.
    """
    if min_value is None and max_value is None:
        return tensor
    elif max_value is not None and min_value is None:
        if isinstance(max_value, float):
            return torch.clamp(tensor, max=max_value)
        else:
            return torch.min(tensor, max_value)
    elif min_value is not None and max_value is None:
        if isinstance(min_value, float):
            return torch.clamp(tensor, min=min_value)
        else:
            return torch.max(tensor, min_value)
    else:
        if isinstance(min_value, numbers.Number) and isinstance(max_value, numbers.Number):
            return torch.clamp(tensor, min=min_value, max=max_value)
        elif isinstance(min_value, float) and isinstance(max_value, torch.Tensor):
            return torch.max(torch.min(tensor, max_value), torch.clamp(tensor, min=min_value))
        elif isinstance(min_value, torch.Tensor) and isinstance(max_value, float):
            return torch.max(torch.min(tensor, torch.clamp(tensor, max=max_value)), min_value)
        else:
            return torch.max(torch.min(tensor, max_value), min_value)


def split_replay_batch(samples):
    """Takes a list of transitions (s, a, r, s') and splits them into arrays.

    Args:
        samples (list): List of tuples (s, a, r, s'), where:
            - s: numpy array, all s must have the same shape.
            - a: numpy array, all a must have the same shape.
            - r: float, the reward.
            - s': numpy array, all non None s' must have the same shape.

    Returns:
        A tuple (states, actions, rewards, next_states, next_states_idx) where each
        element is a numpy array with correspondent data in the rows. Pay attention that the rewards
        array is a column vector.

        Since the next state can be None when episode ends, the number of elements in next_states
        can be < len(samples). For this reason, next_states_idx is a numpy array that contains
        the actual indices of elements in next_states.
    """
    states = np.array([transition[0] for transition in samples])
    actions = np.array([transition[1] for transition in samples])
    rewards = np.array([transition[2] for transition in samples])
    if len(rewards.shape) == 1:
        rewards = np.expand_dims(rewards, 1)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)

    next_states = []
    next_states_idx = []
    for i, transition in enumerate(samples):
        if transition[3] is not None:
            next_states.append(transition[3])
            next_states_idx.append(i)
    next_states = np.array(next_states)
    next_states_idx = np.array(next_states_idx)
    return states, actions, rewards, next_states, next_states_idx


def split_option_replay_batch(samples):
    """Takes a list of transitions (s, o, a, l, r, s') and splits them into arrays.

    Args:
        samples (list): List of tuples (s, a, [is], r, s'), where:
            - s: numpy array, all s must have the same shape.
            - a: numpy array, all a must have the same shape.
            - l: float with the likelihood of the action
            - r: float, the reward.
            - s': numpy array, all non None s' must have the same shape.

    Returns:
        A tuple (states, actions, likelihoods, rewards, next_states, next_states_idx) where each
        element is a numpy array with correspondent data in the rows. Pay attention that the rewards
        array is a column vector.

        Since the next state can be None when episode ends, the number of elements in next_states
        can be < len(samples). For this reason, next_states_idx is a numpy array that contains
        the actual indices of elements in next_states.
    """
    states = np.array([transition[0] for transition in samples])
    options = np.array([transition[1] for transition in samples])
    actions = np.array([transition[2] for transition in samples])
    likelihoods = np.array([transition[3] for transition in samples])
    rewards = np.array([transition[4] for transition in samples])
    if len(rewards.shape) == 1:
        rewards = np.expand_dims(rewards, 1)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)

    next_states = []
    next_states_idx = []
    for i, transition in enumerate(samples):
        if transition[5] is not None:
            next_states.append(transition[5])
            next_states_idx.append(i)
    next_states = np.array(next_states)
    next_states_idx = np.array(next_states_idx)
    return states, options, actions, likelihoods, rewards, next_states, next_states_idx


def split_trace(trace):
    """Split a multi-step trace into a tuple (states, actions, p_actions, rewards).

    Args:
        trace (list): Multi-step transition.

    Returns:
        A tuple (states, actions, p_actions, rewards) where each element is a list of numpy
        arrays, one per time step. The states list contains one state more corresponding to the
        final state of the transition.
    """
    states, actions, p_actions, rewards = [], [], [], []
    t_len = (len(trace) - 1) // 4
    for i in range(t_len):
        t_sub = trace[i * 4: (i + 1) * 4]
        states.append(t_sub[0])
        actions.append(t_sub[1])
        p_actions.append(t_sub[2])
        rewards.append(t_sub[3])
    states.append(trace[-1])
    states = np.array(states)
    actions = np.array(actions)
    p_actions = np.array(p_actions)
    rewards = np.array(rewards)
    return states, actions, p_actions, rewards


def compute_real_targets(episode_rewards, df):
    """Compute the target value at each time step for a given sequence of rewards.

    Args:
        episode_rewards (list): List of rewards from the beginning to the end of the episode.
        df (float): Discount factor.

    Returns:
        A list of the total discounted returns for each time step.
    """
    targets = list(itertools.accumulate(
        episode_rewards[::-1],
        lambda tot, x: x + df * tot
    ))
    return targets[::-1]


def save_models(models, dir):
    os.makedirs(dir, exist_ok=True)
    for name, model in models.items():
        torch.save(model, os.path.join(dir, name))


def load_models(dir, device='cpu'):
    models = {}
    for file in os.listdir(dir):
        models[file] = torch.load(os.path.join(dir, file), device)
    return models


def backbonize(func, act_shape):
    """Take a funciton that maps a state to an action and returns a function that maps a batch of
    state tensors to a batch of action tensors.
    """
    if func is None:
        return None

    def _to_torch(state_batch):
        batch_shape = [state_batch.shape[0]] + [act_shape]
        actions = torch.empty(batch_shape)
        for i, state in enumerate(state_batch):
            action = func(state.detach().numpy())
            actions[i] = torch.tensor(action)
        return actions.float()
    return _to_torch


def state_to_tensor(state):
    return torch.from_numpy(state).float().unsqueeze(0)


def tensor_to_action(action):
    assert action.shape[0] == 1
    return action[0].cpu().numpy()
