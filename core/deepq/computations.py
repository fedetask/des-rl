"""The computations module provides target computers and trainers.

Target computers are objects that take care of computing the target values (e.g. in DQN) that are
generally in the form y = r + max_a' Q(s', a'), but can have different forms (e.g. in Double DQN or
in multi-step returns DQN).

Trainers take care of training the deep Q networks (provided in the deepqnetworks module) from
the given targets.
"""

import abc
import numpy as np
import torch
from torch.nn import functional as F
import numbers


class BaseTargetComputer(abc.ABC):
    """Base class for a target computer, that computes the target values for the given batch and
    DQNetworks.
    """

    @abc.abstractmethod
    def compute_targets(self, batch, *args, **kwargs):
        """Compute the target values for the given batch and Q networks.

        It is left to user responsibility to ensure that the given dq_networks are compatible
        with the derived TargetComputer classes.

        Args:
            batch (list): Sampled batch of transitions. Shape, size, and information of the
                sampled transitions may depend on the ReplayBuffer used, and derivations of this
                class may take that into account.

        Returns:
            Numpy array with the target values.
        """
        pass


class BaseTrainer(abc.ABC):
    """Defines the base class for a Trainer.

    A Trainer takes care of training the DQNetworks from the given target values.
    """

    @abc.abstractmethod
    def train(self, batch, targets, *args, **kwargs):
        """Perform one or several training steps on the DQNetworks object.

        Args:

            batch (tuple): A tuple with all the data needed to train the network. The format of
                the content is left to the derived implementations.
            targets (tuple): Tuple with target values. The format of the content is left to the
                derived implementations.
        """
        pass


class FixedTargetComputer(BaseTargetComputer):
    """Implementation of a computer that computes targets from y = r + max_a' Q(s', a')
    """

    def __init__(self, dq_networks, df=0.99):
        """Instantiate the target computer.

        Args:
            dq_networks (core.deepq.deepqnetworks.BaseDQNetworks): The DQNetworks object that
                will be used to compute the targets.
            df (float): Discount factor in [0, 1].
        """
        self.dq_networks = dq_networks
        self.df = df

    def compute_targets(self, batch, *args, **kwargs):
        states, acts, rewards, next_states, next_states_idx = batch

        target_values = torch.tensor(rewards).float()  # Init target values with rewards
        with torch.no_grad():
            max_q = self.dq_networks.predict_targets(next_states).max(dim=1).values.unsqueeze(dim=1)
            target_values[next_states_idx] += self.df * max_q
        return target_values.numpy()


class DoubleQTargetComputer(BaseTargetComputer):
    """Implementation of the Double Q Network technique for computing the targets.
    """

    def __init__(self, dq_networks, df=0.99):
        """Instantiate the double Q-learning target computer.

        Args:
            dq_networks (core.deepq.deepqnetworks.BaseDQNetworks): The DQNetworks object that
                will be used to compute the targets.
            df (float): The discount factor
        """
        self.dq_networks = dq_networks
        self.df = df

    def compute_targets(self, batch, *args, **kwargs):
        """Compute the one-step target values by, for each sample i in the batch, computing the
        action a_i = argmax Q_phi(s'_i, a'_i) and the targets as y_i = r_i + gamma * Q_phi'(s'_i, a)

        Args:
            batch (tuple): A tuple (states, actions, rewards, next_states, next_states_idx) where
                each element is a numpy array. next_states does not contain next_states for
                states where the episode ended. For this reason, next_states_idx contains indices
                to match next_states with the appropriate elements of states. I.e. if the i-th
                element of next_states_idx is k, then the k-th element of next_states refers to
                the k-th element of states.

        Returns:
            A Numpy column vector with computed target values on the rows.
        """
        states, acts, rewards, next_states, next_states_idx = batch

        target_values = torch.tensor(rewards).float()  # Init target values with rewards

        # Compute Q values with Double Q Learning
        with torch.no_grad():
            # Take actions w.r.t. the learning network
            actions = self.dq_networks.predict_values(next_states).argmax(dim=1).unsqueeze(1)
            # Predict values of those actions with target network
            q_values = self.dq_networks.predict_targets(next_states).gather(dim=1, index=actions)
            target_values[next_states_idx] += self.df * q_values
        return target_values.numpy()


class TD3TargetComputer(BaseTargetComputer):
    """Implementation of a target computer for the TD3 algorithm.

    For more info read the paper https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(self,
                 dqac_nets,
                 max_action=1.,
                 min_action=None,
                 df=0.99,
                 target_noise=0.2,
                 noise_clip=0.5):
        """Instantiate the TD3 target computer.

        Args:
            dqac_nets (core.deepq.deepqnetworks.DeepQActorCritic): The DeepQActorCritic containing
                the target critics and the actor that will be used to compute the target values.
            max_action (Union[float, numpy.ndarray, torch.Tensor]): Right-clipping value for target
                actions after adding the noise. If None, no clipping will be performed.
            min_action (Union[float, numpy.ndarray, torch.Tensor]): Right-clipping value for target
                actions after adding the noise. If None, no clipping will be performed.
            df (float): The discount factor.
            target_noise (Union[float, numpy.ndarray]): Noise added to the target action along
                each dimension.
            noise_clip (Union[float, numpy.ndarray]): Noise will be clipped to
                (-noise_clip, +noise_clip).
        """
        assert type(max_action) == type(min_action), 'max_action and min_action must have same type'

        self.dqac_nets = dqac_nets
        self.max_action = max_action
        self.min_action = min_action
        self.df = df
        self.target_noise = target_noise
        self.noise_clip = noise_clip

        # Conversions of numpy.ndarray to Tensors
        if isinstance(self.max_action, np.ndarray):
            self.max_action = torch.Tensor(self.max_action).float()
            self.min_action = torch.Tensor(self.min_action).float()
        if isinstance(self.target_noise, np.ndarray):
            self.target_noise = torch.Tensor(self.target_noise)
        if isinstance(self.noise_clip, np.ndarray):
            self.noise_clip = torch.Tensor(self.noise_clip).float()

    def compute_targets(self, batch, *args, **kwargs):
        """Compute the targets for the given batch.

        Args:
            batch (tuple): A tuple (states, actions, rewards, next_states, next_states_idx) where
                each element is a numpy array. next_states does not contain next_states for
                states where the episode ended. For this reason, next_states_idx contains indices
                to match next_states with the appropriate elements of states. I.e. if the i-th
                element of next_states_idx is k, then the k-th element of next_states refers to
                the k-th element of states.

        Returns:
            A Numpy column vector with computed target values on the rows.
        """
        states, acts, rewards, next_states, next_states_idx = batch

        states_ts = torch.Tensor(states).float()
        acts_ts = torch.Tensor(acts).float()
        rewards_ts = torch.Tensor(rewards).float()
        next_states_ts = torch.Tensor(next_states).float()
        next_states_idx_ts = torch.Tensor(next_states_idx).long()

        # Sampling and clamping noise
        noise_shape = (len(next_states_idx_ts), *acts_ts.size()[1:])
        noise = (torch.randn(noise_shape) * self.target_noise).float()
        noise = self._clamp(noise, -self.noise_clip, self.noise_clip)

        # Compute target actions and clamping if a max range is specified.
        with torch.no_grad():
            next_action = self.dqac_nets.predict_target_actions(next_states_ts)
            next_action += noise
            if self.max_action is not None:
                next_action = self._clamp(next_action, self.min_action, self.max_action)

            # Compute target values
            target_values = self.dqac_nets.predict_targets(next_states_ts, next_action, mode='min')
            rewards_ts[next_states_idx_ts] += self.df * target_values
        return rewards_ts.detach().numpy()

    def _clamp(self, tensor, min_value, max_value):
        if isinstance(max_value, numbers.Number):
            clamped = tensor.clamp(min_value, max_value)
        elif isinstance(max_value, torch.Tensor):
            clamped = torch.max(
                torch.min(tensor, max_value),
                min_value
            )
        else:
            raise TypeError('The given max_action type ' + str(type(max_value)) +
                            ' is not understood.')
        return clamped


class DQNTrainer(BaseTrainer):

    def __init__(self, dq_networks, loss=F.mse_loss, optimizer=None, lr=1e-4, momentum=0.9):
        """Instantiate the DQNTrainer.

        Args:
            dq_networks (core.algorithms.dqnetworks.BaseDQNetworks): DQNetwork object to be trained.
                Must accept
            loss (function): A function that takes two Tensors (values and targets) and returns a
                scalar tensor with the loss value.
            optimizer (torch.optim.Optimizer): Optimizer to use for training. If None,
                a SGD optimizer with the given learning rate and momentum will be used.
            lr (float): Learning rate for the SGD optimizer.
            momentum (float): Momentum for the SGD optimizer.
        """
        self.dq_networks = dq_networks
        self.loss = loss
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.SGD(
                dq_networks.get_trainable_params(),
                lr=lr,
                momentum=momentum
            )

    def train(self, batch, targets, *args, **kwargs):
        """Perform training steps on the given DQNetworks object for the given inputs and targets.

        Args:
            batch (tuple): A tuple (states, actions, rewards, next_states, next_states_idx) where
                each element is a numpy array. next_states does not contain next_states for
                states where the episode ended. For this reason, next_states_idx contains indices
                to match next_states with the appropriate elements of states. I.e. if the i-th
                element of next_states_idx is k, then the k-th element of next_states refers to
                the k-th element of states.
            targets (numpy.ndarray): Numpy column vector with targets on the rows.

        Returns:
            A tuple with training info: (loss, grads)
        """
        states, actions, rewards, next_states, next_states_idx = batch
        actions_idx_ts = torch.tensor(actions)
        targets_ts = torch.tensor(targets).float()
        values = self.dq_networks.predict_values(states).gather(1, actions_idx_ts)
        loss = self.loss(values, targets_ts)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return (loss.detach().numpy(), [p.grad.numpy() for p in
                                        self.dq_networks.get_trainable_params()])


class TD3Trainer(BaseTrainer):
    """Implementation of a trainer according to the TD3 algorithm.

    For more info read the paper https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(self,
                 dqac_networks,
                 loss=F.mse_loss,
                 critic_optimizer=None,
                 actor_optimizer=None,
                 critic_lr=0.5e-3,
                 actor_lr=0.5e-3):
        """Instantiate the TD3 trainer.

        Args:
            dqac_networks (core.deepq.deepqnetworks.DeepQActorCritic): The DeepQActorCritic
                containing the networks to train.
            loss (function):
                loss (function): A function that takes two Tensors (values and targets) and returns
                a scalar tensor with the loss value.
            critic_optimizer (torch.optim.Optimizer): Optimizer to use for training the critics.
                If None, an Adam optimizer with the given learning rate will be used.
            actor_optimizer (torch.optim.Optimizer): Optimizer to use for training the actor.
                If None, an Adam optimizer with the given learning rate will be used.
            critic_lr (float): Learning rate for the critic optimizer when using the default.
            actor_lr (float): Learning rate for the actor optimizer when using the default.
        """
        self.dqac_networks = dqac_networks
        self.loss = loss
        if critic_optimizer is None:
            self.critic_optimizer = torch.optim.Adam(dqac_networks.get_trainable_params()[0],
                                                     lr=critic_lr)
        if actor_optimizer is None:
            self.actor_optimizer = torch.optim.Adam(dqac_networks.get_trainable_params()[1],
                                                    lr=actor_lr)

    def train(self, batch, targets, train_actor=False, *args, **kwargs):
        """Perform one optimization step on the critic and actor networks for the given samples.

        Args:
            batch (tuple): A tuple (states, actions, rewards, next_states, next_states_idx) where
                each element is a numpy array. next_states does not contain next_states for
                states where the episode ended. For this reason, next_states_idx contains indices
                to match next_states with the appropriate elements of states. I.e. if the i-th
                element of next_states_idx is k, then the k-th element of next_states refers to
                the k-th element of states.
            targets (numpy.ndarray): A numpy array with target values for each state-action pair in
                the batch.
            train_actor (bool): Whether to train the actor network.

        """
        states, actions, rewards, next_states, next_states_idx = batch
        states_ts = torch.Tensor(states).float()
        actions_ts = torch.Tensor(actions).float()
        targets_ts = torch.Tensor(targets).float()

        q_values = self.dqac_networks.predict_values(states_ts, actions_ts, mode='all')
        critic_loss = q_values.size()[1] * self.loss(q_values, targets_ts[:, None, :])
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if train_actor:
            act = self.dqac_networks.predict_actions(states_ts)
            actor_loss = - self.dqac_networks.predict_values(states_ts, act).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


