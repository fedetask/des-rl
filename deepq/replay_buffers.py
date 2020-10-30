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
            A tuple (transitions, info) where transitions is a list of sampled transitions, and
            info is a dictionary with additional information.
        """
        pass


class FIFOReplayBuffer(BaseReplayBuffer):
    """Defines a simple fixed-size, FIFO evicted replay buffer, that stores transitions and allows
    sampling.

    Transitions are tuples in the form (s, a, r, s', ...), where after s' any additional
    information can be stored.
    """

    def __init__(self, maxlen):
        """Instantiate the replay buffer.

        Args:
            maxlen (int): Maximum number of transitions to be stored.
        """
        super().__init__()
        self.buffer = collections.deque(maxlen=maxlen)

    def remember(self, transition, *args, **kwargs):
        """Store the given transition
        Args:
            transition (list): List in the form [s, a, r, s', ...]. Note that s' should be None
                if the episode ended. After s' any additional information can be passed.

        Raises:
            AssertionError if given shapes do not match with those declared at initialization.
        """
        self.buffer.append(transition)

    def sample(self, size, *args, **kwargs):
        """Sample uniformly from the replay buffer.

        Args:
            size (int): Number of transitions to sample.

        Returns:
            A tuple (transitions, info). transitions is a list with the sampled transitions.
            info is an empty dictionary.
        """
        if size > len(self.buffer):
            raise ValueError(
                'Trying to sample ' + str(size) + ' items when buffer has only ' +
                str(len(self.buffer)) + ' items.'
            )

        indices = np.arange(len(self.buffer))
        sampled_indices = np.random.choice(a=indices, size=size, replace=False)
        return [self.buffer[i] for i in sampled_indices], {}  # Empty dict for compatibility


class PrioritizedReplayBuffer(FIFOReplayBuffer):
    """Implementation of a prioritized replay buffer.

    This replay buffer stores transitions (s, a, r, s', w) where w is the weight. The sampling is
    done by sampling a chunk of the given chunk_size, and performing a weighted sampling on it.
    This allows sampling to be done in constant time. The probability of a transition i is given
    by w_i^alpha / sum w_k^alpha. If exceeding the maximum length, samples are evicted with a
    FIFO policy.
    """

    def __init__(self, maxlen, alpha=0.8, chunk_size=2000):
        """Instantiate the replay buffer.

        Args:
            maxlen (int): Maximum number of transitions that the replay buffer will keep.
            alpha (float): Level of prioritization, between 0 and 1.
            chunk_size (int): Dimension of the random chunk from which transitions will be sampled.
        """
        super().__init__(maxlen=maxlen)
        self.alpha = alpha
        self.chunk_size = chunk_size
        self.avg_td_error = 0

    def remember(self, transition, *args, **kwargs):
        """Add the transition to the buffer.

        Args:
            transition (list): Transition in the form [s, a, r, s', w, ...] where w is the
                un-normalized weight of the transition.
        """
        assert len(transition) >= 5, 'The given transition must be [s, a, r, s\', w, ...].'
        super().remember(transition)

    def sample(self, size, *args, **kwargs):
        """Sample the given number of transitions with probability proportional to the weights.

        Args:
            size (int): Number of transitions to sample.

        Returns:
            A tuple (transitions, info) where transitions is a list of the sampled transitions and
            info is a dictionary {'weights':  weights} with weights being the normalized weights
            of the sampled transitions.
        """
        if size > len(self.buffer):
            raise ValueError(
                'Trying to sample ' + str(size) + ' items when buffer has only ' +
                str(len(self.buffer))
            )

        chunk = np.random.choice(a=np.arange(len(self.buffer)), size=self.chunk_size, replace=False)
        td_errors = np.array([self.buffer[i][4] + 1e-6 for i in chunk])
        self.avg_td_error = td_errors.mean()  # Update statistics
        # Compute probabilities and sample
        probabilities = np.power(td_errors, self.alpha)
        probabilities /= np.sum(probabilities)
        sampled_indices = np.random.choice(
            a=range(len(chunk)), size=size, p=probabilities, replace=False
        )
        sampled = [self.buffer[i] for i in chunk[sampled_indices]]
        weights = np.power(len(self.buffer) * probabilities[sampled_indices], -1)
        weights /= np.sum(weights)
        return sampled, {'weights': weights}


# ----------------------------------------- PREFILLERS --------------------------------------------#


class BufferPrefiller:
    """Prefiller that adds transitions to the replay buffer by sampling random actions from a Gym
    environment.
    """

    def __init__(self, num_transitions, add_info=False, shuffle=False, prioritized_replay=False,
                 collection_policy=None, collection_policy_noise=None, min_action=None,
                 max_action=None, use_residual=False):
        """Instantiate the buffer prefiller.

        Args:
            num_transitions (int): Number of transitions to be added to the replay buffer.
            add_info (bool): Whether to append the additional information to the transitions.
            shuffle (bool): Whether to shuffle the replay buffer after sampling the given number
                of transitions.
            prioritized_replay (bool): Whether to add an additional element w to the sampled
                transitions for prioritized replay buffers. w is set as 1 / num_transitions.
                if prioritized_replay is True, the transition will be (s, a, r, s', w, [info]).
            collection_policy: Function (numpy.ndarray) -> numpy.ndarray that returns an action
                for a given state. Will be used to collect transitions.
            collection_policy_noise (float): Standard deviation of noise added to actions
                selected by the collection_policy.
        """
        self.num_transitions = num_transitions
        self.add_info = add_info
        self.shuffle = shuffle
        self.prioritized_replay = prioritized_replay
        self.collection_policy = collection_policy
        self.collection_policy_noise = collection_policy_noise
        self.min_action = min_action
        self.max_action = max_action
        self.use_residual = use_residual
        if collection_policy_noise is not None:
            assert min_action is not None and max_action is not None,\
                'Min and max action bust be specified when adding collection policy noise.'
        else:
            assert not use_residual and collection_policy_noise is None, \
                'use_residual and collection_policy_noise can only be used together with a ' \
                'collection policy.'

    def fill(self, replay_buffer, env):
        """Add the given number of transitions to the replay buffer by sampling
        random actions in the given environment.

        Args:
            replay_buffer (BaseReplayBuffer): A replay buffer implementation.
            env (gym.core.Env): A Gym environment.
        """
        s = env.reset()
        for step in range(self.num_transitions):
            if self.collection_policy is None:
                a = env.action_space.sample()
            else:
                a = self.collection_policy(s)
                if self.collection_policy_noise is not None:
                    noise = np.random.normal(
                        0, self.collection_policy_noise, env.action_space.shape)
                    a = (a + noise).clip(self.min_action, self.max_action)
            s_prime, r, done, info = env.step(a)
            s_prime = s_prime if not done else None

            if self.use_residual:
                transition = [s, noise, r, s_prime]
            else:
                transition = [s, a, r, s]
            if self.prioritized_replay:
                transition.append(1. / self.num_transitions)
            if self.add_info:
                transition.append(info)
            replay_buffer.remember(transition)
            if done:
                s = env.reset()
            else:
                s = s_prime
        if self.shuffle:
            random.shuffle(replay_buffer)
