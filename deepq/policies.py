"""This module contains base classes and implementations for Deep Q policies, i.e. policies that
choose from a list of Q values. Any policy implementation must have BasePolicy as superclass.

Content:
    Base classes:
        - BasePolicy: base class for any policy that selects an action from Q values
        - EpsilonGreedyPolicy: base class for epsilon-greedy policies.
        - BaseGaussianPolicy: base class for continuous policies modeled by a multivariate Gaussian
            distribution over the network outputs.
        - BaseEpsilonGaussianPolicy: base class for Gaussian policies that add noise to the
            predicted actions.
    Implementations:
        - FixedEpsilonGreedyPolicy
        - LinearDecayEpsilonGreedyPolicy
        - ExponentialDecayEpsilonGreedy
        - FixedEpsilonGaussianPolicy
        - LinearDecayEpsilonGaussianPolicy
        - ExponentialDecayEpsilonGaussianPolicy
"""

import abc
import sys
import os

import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common


class BasePolicy(abc.ABC):
    """Base class for a policy, that defines the minimal interface that any policy should present.
    """

    @abc.abstractmethod
    def act(self, q_values, *args, **kwargs):
        """Return the action to take for the given state.

        Args:
            q_values (numpy.ndarray): Numpy array with Q values for each of the possible actions.

        Returns:
            A numpy array with the index (or indices) of the chosen action(s).
        """
        pass


class EpsilonGreedyPolicy(BasePolicy):
    """Base class for an epsilon-greedy policy, that selects the argmax action with probability
    p = 1 - epsilon and a random action with p = epsilon.
    """

    def __init__(self, start_epsilon, end_epsilon, decay_steps, decay_schedule='lin', probs=None):
        """Instantiate the base epsilon greedy policy.

        Args:
            start_epsilon (float): Initial value of epsilon.
            end_epsilon (float): Final value of epsilon.
            decay_steps (int): Number of steps over which epsilon will be decayed from
                start_epsilon to end_epsilon.
            decay_schedule (str): Schedule for decaying epsilon, 'const', 'lin', 'exp'.
        """
        self.epsilon_updater = common.ParameterUpdater(
            start_epsilon, end_epsilon, decay_steps, decay_schedule)
        self.probs = probs

    def act(self, q_values, *args, **kwargs):
        """Choose the action from the given Q values, update epsilon, and return the action.

        Args:
            q_values (numpy.ndarray): Numpy array with Q values for each of the possible actions.

        Returns:
            A 1-dimensional numpy array with the index of the chosen action.
        """
        if np.random.binomial(1, p=self.epsilon_updater.cur_value):
            if self.probs is None:
                action = np.array([np.random.choice(range(len(q_values)))])
            else:
                action = np.array([np.random.choice(range(len(q_values)), p=self.probs)])
        else:
            action = np.array([np.argmax(q_values)])
        self.epsilon_updater.update()
        return action


class GreedyPolicy(BasePolicy):
    """Greedy policy that returns the action for which the Q values are higher.
    """

    def act(self, q_values, *args, **kwargs):
        """Return the argmax over the Q values

        Args:
            q_values (numpy.ndarray): Numpy array with Q values for each of the possible actions.

        Returns:
            A 1-dimensional numpy array with the index of the action that maximizes the Q values.
        """
        return np.array([np.argmax(q_values)])


class BaseGaussianPolicy(abc.ABC):
    """Base class for policies that sample from a Gaussian distribution with mean given by the
    predicted action.
    """

    @abc.abstractmethod
    def act(self, mean, *args, **kwargs):
        """Sample an action from a Gaussian distribution with the predicted mean.

        Args:
            mean (torch.Tensor): Mean of the Gaussian.

        Returns:
            A Tensor with the same shape of mean, containing the chosen action.
        """
        pass


class BaseEpsilonGaussianPolicy(BaseGaussianPolicy):
    """Base class for a parametrized Gaussian policy that adds noise from a standard distribution
    with zero mean and variance epsilon, which is updated over act() calls with the given schedule.
    """

    def __init__(self, start_epsilon, end_epsilon, decay_steps, decay_schedule='const',
                 min_action=None, max_action=None):
        """Instantiate the Gaussian policy.

        Args:
            start_epsilon (float): Initial variance of the distribution.
            end_epsilon (float): Final variance of the distribution.
            decay_steps (int): Number of steps over which epsilon is updated from start_epsilon
                to end_epsilon.
            decay_schedule (str): 'const', 'lin', 'exp'.
            min_action (Union[float, numpy.ndarray]): Low clipping value of sampled actions.
                Leave None for no clipping.
            max_action (Union[float, numpy.ndarray]): High clipping value of sampled actions.
                Leave None for no clipping.
        """
        self.min_action = min_action
        self.max_action = max_action
        self.epsilon_updater = common.ParameterUpdater(
            start_epsilon, end_epsilon, decay_steps, decay_schedule)

    def act(self, mean, *args, **kwargs):
        """Compute the action by sampling from a standard Gaussian with zero mean and epsilon
        variance.

        Args:
            mean (torch.Tensor): The predicted action, that will be used as mean.

        Returns:
            A Tensor with the same shape of mean, containing the chosen action.
        """
        noise = torch.randn_like(mean) * np.sqrt(self.epsilon_updater.cur_value)
        action = mean + noise
        action = common.clamp(action, self.min_action, self.max_action)
        self.epsilon_updater.update()
        return action
