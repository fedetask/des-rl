"""This module defines the replay buffers. A replay buffer is a data structure that stores
transitions coming from the environment, and allows sampling. This module provides a base class
BaseReplayBuffer that defines the minimal interface that any replay buffer implementation should
provide.

Contents:
    Base classes:
        - BaseReplayBuffer
    Implementations:
        -FIFOReplayBuffer
"""

import collections
import random
import abc
import numbers

import numpy as np


# --------------------------------------- REPLAY BUFFERS ------------------------------------------#


class BaseReplayBuffer(abc.ABC, collections.UserList):
    """The base class for replay buffers.

    Any derived replay buffer must present an Iterable interface, therefore allowing iteration,
    sampling, etc.
    """

    def add_iterable(self, iterable):
        for i in iterable:
            self.remember(i)

    def __add__(self, e):
        if hasattr(e, '__iter__'):
            return self.add_iterable(e)
        else:
            return self.remember(e)

    def append(self, e):
        self.remember(e)

    def extend(self, e):
        return self.add_iterable(e)

    @abc.abstractmethod
    def remember(self, transition, *args, **kwargs):
        """Remember the given transition.

        Args:
            transition (tuple): A transition in the form (s, a, r, s', *info). After s' any
            additional information can be passed.
        """
        pass

    @abc.abstractmethod
    def sample(self, size, *args, **kwargs):
        """Sampling operation on the replay buffer.

        Args:
            size (int): Number of transitions to sample.

        Returns:
            A tuple in the form (states, actions, rewards, next_states, ...), where each element is
            a numpy array with the correspondent data in the rows. After next_states,
            any additional information can be returned.
        """
        pass


class FIFOReplayBuffer(BaseReplayBuffer):
    """Defines a simple fixed-size, FIFO evicted replay buffer, that stores transitions and allows
    sampling.

    Transitions are tuples in the form (s, a, r, s', ...), where after s' any additional
    information can be stored.
    """

    def __init__(self, maxlen, state_shape, action_shape):
        """Instantiate the replay buffer.

        Args:
            maxlen (int): Maximum number of transitions to be stored.
            state_shape (tuple): Shape of a state.
            action_shape (tuple): Shape of an action.
        """
        super().__init__()
        self.buffer = collections.deque(maxlen=maxlen)
        self.state_shape = state_shape
        self.action_shape = action_shape

        if isinstance(self.state_shape, int):
            self.state_shape = (self.state_shape, )
        if isinstance(self.action_shape, int):
            self.action_shape = (self.action_shape, )

    def remember(self, transition, *args, **kwargs):
        """Store the given transition
        Args:
            transition (tuple): Tuple in the form (s, a, r, s', ...). Note that s' should be None
                if the episode ended. After s' any additional information can be passed.

        Raises:
            AssertionError if given shapes do not match with those declared at initialization.
        """
        s, a, r, s_prime, *_ = transition
        assert isinstance(s, np.ndarray) and s.shape == self.state_shape,\
            'The given state ' + str(s) + ' does not match shape ' + str(self.state_shape)
        if isinstance(a, np.ndarray):
            assert a.shape == self.action_shape, \
                'The given action ' + str(a) + ' does not match shape ' + str(self.action_shape)
        elif self.action_shape[0] == 1 and isinstance(a, numbers.Number):
            a = np.array([a])  # For simplicity actions are always numpy.ndarray
        else:
            raise AssertionError('The given action ' + str(a) + ' does not match shape ' + str(
                self.action_shape))

        if s_prime is not None and isinstance(s_prime, np.ndarray):
            assert s_prime.shape == self.state_shape, \
                'The given state ' + str(s_prime) + ' does not match shape ' + str(self.state_shape)

        self.buffer.append((s, a, r, s_prime))

    def sample(self, size, *args, **kwargs):
        """Sample uniformly from the replay buffer.

        Args:
            size (int): Number of transitions to sample.

        Returns:
            A list of the sampled transitions.
        """
        if size > len(self.buffer):
            raise ValueError('Trying to sample ' + str(size) + ' items when buffer has only ' +
                             str(len(self.buffer)))

        indices = np.arange(len(self.buffer))
        sampled_indices = np.random.choice(a=indices, size=size)
        return [self.buffer[i] for i in sampled_indices]


class PrioritizedReplayBuffer(FIFOReplayBuffer):
    """Implementation of a prioritized replay buffer.

    This replay buffer stores transitions (s, a, r, s', w) where w is the weight. Transitions are
    sampled with probabilities proportional to this weights.
    """

    def remember(self, transition, *args, **kwargs):
        """Add a transition to the replay buffer.

        The weight of the transition is normalized by dividing it by the sum of weights in the
        buffer remember() cost is therefore O(buffer length). TODO: Improve efficiency

        Args:
            transition (tuple): A tuple like (s, a, r, s', w). w is the weight and must be >= 0.
        """
        s, a, r, s_prime, w = transition
        error_str = 'Weights of a transitions must be postive numbers.'
        assert isinstance(w, numbers.Number), error_str
        assert w >= 0, error_str

        norm_weight = w / sum(t[4] for t in self.buffer)
        super().remember((s, a, r, s_prime, norm_weight))

    def sample(self, size, *args, **kwargs):
        """Sample the given number of transitions with probability proportional to the weights.

        Args:
            size (int): Number of transitions to sample.

        Returns:
            A list of the sampled transitions.
        """
        if size > len(self.buffer):
            raise ValueError('Trying to sample ' + str(size) + ' items when buffer has only ' +
                             str(len(self.buffer)))

        indices = np.arange(len(self.buffer))
        weights = np.array([t[4] for t in self.buffer])
        sampled_indices = np.random.choice(a=indices, size=size, p=weights)
        return [self.buffer[i] for i in sampled_indices]


# ----------------------------------------- PREFILLERS --------------------------------------------#


class UniformGymPrefiller:
    """Prefiller that adds transitions to the replay buffer by sampling random actions from a Gym
    environment.
    """

    def fill(self, replay_buffer, env, num_transitions, add_info=False, shuffle=False):
        """Add the given number of transitions to the replay buffer by sampling
        random actions in the given environment.

        A transition is intended in the form (s, a, r, s', [info]) where s' is None if the episode
        ended. Remember to call reset() on the environment after using this function.

        Args:
            replay_buffer (BaseReplayBuffer): A replay buffer implementation.
            env (gym.core.Env): A Gym environment.
            num_transitions (int): Number of transitions to be added to the replay buffer.
            add_info (bool): Whether to append the additional information to the transitions.
            shuffle (bool): Whether to shuffle the replay buffer after sampling the given number
                of transitions.
        """
        state = env.reset()
        for step in range(num_transitions):
            a = env.action_space.sample()
            s_prime, r, done, info = env.step(a)
            if done:
                s_prime = None
            transition = (state, a, r, s_prime) if not add_info else (state, a, r, s_prime, info)
            replay_buffer.remember(transition)
            if done:
                state = env.reset()
            else:
                state = s_prime
        if shuffle:
            random.shuffle(replay_buffer)
