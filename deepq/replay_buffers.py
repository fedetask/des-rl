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
import torch

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


class Trajectory:
    """Container class for a trajectory.

    A trajectory is a sequence (s_0, a_0, p_a_0, r_0, s_1, ...). A trajectory ends with a next
    state s_n if the episode is not over. The episode_ended() method can be used to check whether
    the last state in states corresponds to a terminal state or not.
    """

    def __init__(self):
        self.states = []  # Note that states[1:] corresponds to the next states
        self.actions = []
        self.p_actions = []  # Policy for the state-action pair at the time the action was taken.
        self.rewards = []
        self.last_state = None  # Final state if episode didn't end

    def episode_ended(self):
        """Return True if the this trajectory ended because of an episode termination,
        false otherwise.

        If True, then states has the same length as the other attributes, as there is no next state.
        If False, states has an additional state corresponding to the state at which the
        trajectory was cut.
        """
        return self.last_state is None

    def length(self):
        """Returns the length of the trajectory, without counting the possible additional final
        state.
        """
        return len(self.actions)

    def truncate(self, start, end):
        """Return a copy of this trajectory, truncated from the given start and end indices.

        The states list will contain the additional next state unless the trajectory ends at
        index end due to episode termination.

        Args:
            start (int): Index from which to keep transitions.
            end (int): Last index (inclusive) of the kept transitions.
        """
        new_t = Trajectory()
        new_t.states = self.states[start:end + 1]
        new_t.actions = self.actions[start: end + 1]
        new_t.p_actions = self.p_actions[start: end + 1]
        new_t.rewards = self.rewards[start: end + 1]
        if len(self.states) > end + 1:
            new_t.last_state = self.states[end+1]
        return new_t


class EpisodicReplayBuffer(BaseReplayBuffer):
    """Implementation of a replay buffer that stores Trajectory elements.
    """

    def __init__(self, maxlen, min_trajectory_len=2):
        super().__init__()
        self.maxlen = maxlen
        self.min_trajectory_len = min_trajectory_len
        self.buffer = collections.deque(maxlen=maxlen)
        self._cur_trajectory = Trajectory()

    def remember(self, transition, *args, **kwargs):
        """Append a transition in the form (s, a, p_a, r, done) to the current cached
        trajectory.

        If done is True, s' is added to the cached trajectory, which is then added to the buffer
        and a new empty one is instantiated.

        Args:
            transition (tuple): A tuple (s, a, p_a, r, done).
        """
        self._cur_trajectory.states.append(transition[0])
        self._cur_trajectory.actions.append(transition[1])
        self._cur_trajectory.p_actions.append(transition[2])
        self._cur_trajectory.rewards.append(transition[3])
        if transition[4]:  # If done
            self._store_and_reset()

    def cutoff(self, next_state):
        """Signal the replay buffer that the current cached trajectory has been cut by the
        algorithm.

        The given next state is added to the cached trajectory, which is then stored in the
        replay buffer. Do not call this if the episode terminates, as add_transition() already
        deals with it.

        Args:
            next_state (torch.Tensor): A Tensor containing the state at which the trajectory
                was cut.
        """
        self._cur_trajectory.states.append(next_state)
        self._store_and_reset()

    def sample(self, batch_size, random_start=False, same_length=False):
        """Return a list of batch_size Trajectory objects sampled uniformly from the buffer and
        truncated to have the same length.

        Args:
            batch_size (int): Number of trajectories to sample.
            same_length (bool): Whether to cut trajectories to have the same length.
            random_start (bool): Whether the initial step of each trajectory is sampled randomly.
        """
        assert len(self.buffer) >= batch_size, \
            f'Cannot sample {batch_size} trajectories from buffer of length {len(self.buffer)}.'
        indices = np.random.choice(range(len(self.buffer)), size=batch_size, replace=False)
        trajectories = [self.buffer[i] for i in indices]
        if not random_start and not same_length:
            return trajectories

        if random_start:
            start_indices = [np.random.choice(range(int(t.get_length()))) for t in trajectories]
        else:
            start_indices = [0] * len(trajectories)
        if same_length:
            min_len = min(t.len() - start_indices[i] for i, t in enumerate(trajectories))
            end_indices = [start_indices[i] + min_len - 1 for i, t in enumerate(trajectories)]
        else:
            end_indices = [t.get_length() - 1 for t in trajectories]
        res_trajectories = [
            t.truncate(start_indices[i], end_indices[i]) for i, t in enumerate(trajectories)]
        return res_trajectories

    def n_steps(self):
        """Returns the sum of lengths of trajectories in the buffer.
        """
        return sum(t.get_length() for t in self.buffer)

    def length(self):
        """Return the number of trajectories contained in the replay buffer.
        """
        return len(self.buffer)

    def _store_and_reset(self):
        if len(self._cur_trajectory.actions) >= self.min_trajectory_len:
            self.buffer.append(self._cur_trajectory)
        self._cur_trajectory = Trajectory()


# ----------------------------------------- PREFILLERS --------------------------------------------#


class BufferPrefiller:
    """Prefiller that adds transitions to the replay buffer by sampling random actions from a Gym
    environment.
    """

    def __init__(self, num_transitions, add_info=False, shuffle=False, prioritized_replay=False,
                 collection_policy=None, min_action=None, max_action=None, use_residual=False):
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
        """
        self.num_transitions = num_transitions
        self.add_info = add_info
        self.shuffle = shuffle
        self.prioritized_replay = prioritized_replay
        self.collection_policy = collection_policy
        self.min_action = min_action
        self.max_action = max_action
        self.use_residual = use_residual

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
                with torch.no_grad():
                    a = self.collection_policy(torch.from_numpy(s).float().unsqueeze(0))[0]\
                        .detach().numpy()
                    noise = np.random.uniform(self.min_action - a, self.max_action - a)
                    a = a + noise
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
