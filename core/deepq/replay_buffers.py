import collections
import random
import abc
import numbers

import numpy as np


class BaseReplayBuffer(abc.ABC):
    """The base class for replay buffers.

    Any derived replay buffer must present an Iterable interface, therefore allowing iteration,
    sampling, etc.
    """

    @abc.abstractmethod
    def remember(self, transition, *args, **kwargs):
        """Remember the given transition.

        Args:
            transition (list): A transition in the form (s, a, r, s', *info). After s' any
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
        samples = random.sample(self.buffer, size)
        return samples
