"""This module contains base classes and implementations for Deep Q policies, i.e. policies that
choose from a list of Q values. Any policy implementation must have BasePolicy as superclass.

Content:
    Base classes:
        - BasePolicy: base class for any policy.
        - EpsilonGreedyPolicy: base class for epsilon-greedy policies.
    Implementations:
        - FixedEpsilonGreedyPolicy
        - LinearDecayEpsilonGreedyPolicy
        - ExponentialDecayEpsilonGreedy
"""

import abc

import numpy as np


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
    """Base class for an epsilon-greedy policy.

    Implementations of this class should use _update_params() to handle parameter updating every
    time act() is called.
    """

    def __init__(self, epsilon):
        """Instantiate the base epsilon greedy policy.

        Args:
            epsilon (float): Probability of taking a random action.
        """
        self.cur_epsilon = epsilon

    def act(self, q_values, *args, **kwargs):
        """Choose the action from the given Q values, update epsilon, and return the action.

        Args:
            q_values (numpy.ndarray): Numpy array with Q values for each of the possible actions.

        Returns:
            A 1-dimensional numpy array with the index of the chosen action.
        """
        rand_action = np.random.choice([True, False], p=[self.cur_epsilon, 1 - self.cur_epsilon])
        if rand_action:
            action = np.array([np.random.choice(range(len(q_values)))])
        else:
            action = np.array([np.argmax(q_values)])
        self._update_params()
        return action

    @abc.abstractmethod
    def _update_params(self):
        """Update the policy parameters.

        This is called in every execution of act() and should implement any parameter update that
        happens at every act() call.
        """
        pass


# ----------------------------------------- IMPLEMENTATIONS ------------------------------------ #


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


class FixedEpsilonGreedyPolicy(EpsilonGreedyPolicy):
    """An epsilon-greedy policy that takes a random action with p = epsilon, and an argmax action
    with p = (1 - epsilon). This policy only works for single actions and not for multi-discrete
    actions.
    """

    def _update_params(self):
        """Does nothing, since epsilon is fixed.
        """
        pass


class LinearDecayEpsilonGreedyPolicy(EpsilonGreedyPolicy):
    """An epsilon-greedy policy where the epsilon parameter decays linearly with the number of
    act() calls. Works only with single-action and not for multi-discrete actions.
    """

    def __init__(self, decay_steps, start_epsilon, end_epsilon):
        """Instantiate the linear decay epsilon-greedy policy with the linear decay parameters.

        Args:
            decay_steps (int): Number of act calls necessary to make epsilon decay from
                start_epsilon to end_epsilon.
            start_epsilon (float): Initial value of epsilon.
            end_epsilon (float): Final value of epsilon.
        """
        super().__init__(epsilon=start_epsilon)
        self.decay_steps = decay_steps
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.step = 0

    def _update_params(self):
        """Linearly decay epsilon, silently setting it at zero if it becomes negative.
        """
        # Epsilon update
        self.cur_epsilon = self.cur_epsilon - \
                           (self.start_epsilon - self.end_epsilon) / self.decay_steps
        if self.cur_epsilon < 0:
            self.cur_epsilon = 0

    def reset(self):
        """Reset the decay parameters to their initial value
        """
        self.cur_epsilon = self.start_epsilon
        self.step = 0


class ExponentialDecayEpsilonGreedy(EpsilonGreedyPolicy):
    """Implementation of an exponential decay epsilon-greedy policy.

    The policy takes a random action with p = epsilon and argmax action over the Q values with
    p = (1 - epsilon). Epsilon exponentially decays with the number of steps, that correspond to
    the number of act() calls. The formula is epsilon(t) = start_epsilon * alpha ** t,
    where alpha is computed such that start_epsilon * alpha ** decay_steps = end_epsilon.
    """

    def __init__(self, decay_steps, start_epsilon, end_epsilon):
        """Instantiate the exponential decay epsilon-greedy policy.

        Args:
            decay_steps (int): Number of steps to make epsilon decay to end_epsilon.
            start_epsilon (float): Initial value of epsilon.
            end_epsilon (float): Final value of epsilon.
        """
        super().__init__(start_epsilon)
        self.decay_steps = decay_steps
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon

        if end_epsilon == 0:
            end_epsilon = 1e-4
        self.alpha = np.power((end_epsilon / start_epsilon), 1./decay_steps)
        self.step = 0

    def _update_params(self):
        """Perform one exponential decay step
        """
        self.cur_epsilon *= self.alpha
