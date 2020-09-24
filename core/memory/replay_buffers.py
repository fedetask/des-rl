import collections
import random
import abc


class BaseReplayBuffer(abc.ABC):
    """The base class for replay buffers.

    Any derived replay buffer must present an Iterable interface, therefore allowing iteration,
    sampling, etc.
    """

    @abc.abstractmethod
    def remember(self, transition, *args, **kwargs):
        """Remember the given transition.

        Args:
            transition (list): A transition in the form (s, a, r, s', ...). After s' any additional
                information can be passed.
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

    Transitions are tuples in the form (s, a, r, s', *additional_info), where additional_info may
    be a list with any additional information.
    """

    def __init__(self, maxlen, state_shape, action_shape):
        """Instantiate the replay buffer.

        Args:
            maxlen (int): Maximum number of transitions to be stored.
            state_shape (tuple): Shape of states.
            action_shape (tuple): Shape of actions.
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
            transition (tuple): Tuple in the form (s, a, r, s', *additional_info).

        Raises:
            AssertionError if s, a, or s' shapes do not match with those declared at initialization.
        """
        assert transition[0].shape == self.state_shape
        assert transition[1].shape == self.action_shape
        if transition[3] is not None:
            assert transition[3].shape == self.state_shape
        self.buffer.append(transition)

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
