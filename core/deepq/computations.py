"""The computations module provides target computers and trainers.

Target computers are objects that take care of computing the target values (e.g. in DQN) that are
generally in the form y = r + max_a' Q(s', a'), but can have different forms (e.g. in Double DQN or
in multi-step returns DQN).

Trainers take care of training the deep Q networks (provided in the deepqnetworks module) from
the given targets.
"""

import abc

import torch
from torch.nn import functional as F


class BaseTargetComputer(abc.ABC):
    """Base class for a target computer, that computes the target values for the given batch and
    DQNetworks.
    """

    @abc.abstractmethod
    def compute_targets(self, dq_networks, batch, *args, **kwargs):
        """Compute the target values for the given batch and Q networks.

        It is left to user responsibility to ensure that the given dq_networks are compatible
        with the derived TargetComputer classes.

        Args:
            dq_networks (BaseDQNetworks): The DQNetwork implementation that contains the QNetworks
                that will be used to compute the targets
            batch (list): Sampled batch of transitions. Shape, size, and information of the
                sampled transitions may depend on the ReplayBuffer used, and derivations of this
                class may take that into account.

        Returns:
            Numpy array with the target values.
        """
        pass


class FixedTargetComputer(BaseTargetComputer):
    """Implementation of a computer that computes targets from y = r + max_a' Q(s', a')
    """

    def __init__(self, df):
        """Instantiate the target computer.

        Args:
            df (float): Discount factor in [0, 1].
        """
        self.df = df

    def compute_targets(self, dq_networks, batch, *args, **kwargs):
        states, acts, rewards, next_states, next_states_idx = batch

        target_values = torch.tensor(rewards).float()  # Init target values with rewards
        with torch.no_grad():
            max_q = dq_networks.predict_targets(next_states).max(dim=1).values.unsqueeze(dim=1)
            target_values[next_states_idx] += self.df * max_q
        return target_values.numpy()


class DoubleQTargetComputer(BaseTargetComputer):
    """Implementation of the Double Q Network technique for computing the targets.
    """

    def __init__(self, df):
        """Instantiate the double Q-learning target computer.

        Args:
            df (float): The discount factor
        """
        self.df = df

    def compute_targets(self, dq_networks, batch, *args, **kwargs):
        """Compute the one-step target values by, for each sample i in the batch, computing the
        action a_i = argmax Q_phi(s'_i, a'_i) and the targets as y_i = r_i + gamma * Q_phi'(s'_i, a)

        Args:
            dq_networks (BaseDQNetworks): A DQNetworks implementation.
            batch (tuple): A tuple (states, actions, rewards, next_states, next_states_idx) where:
                - states (ndarray): Numpy array with states on the rows.
                - actions (ndarray): Numpy array with actions correspondent to states on the rows.
                - rewards (ndarray): Numpy column vector with rewards correspondent to states and
                    actions on the rows
                - next_states (ndarray): Numpy array with next states on the rows. This may be
                    shorter than states, actions and rewards if the episode ended.
                - next_states_idx (ndarray): Numpy array containing indices to match next_states
                    with the appropriate elements of states. I.e. if the i-th element of
                    next_states_idx is k, then the k-th element of next_states refers to the k-th
                    element of states, actions, rewards.

        Returns:
            A Numpy column vector with computed target values on the rows.
        """
        states, acts, rewards, next_states, next_states_idx = batch

        target_values = torch.tensor(rewards).float()  # Init target values with rewards

        # Compute Q values with Double Q Learning
        with torch.no_grad():
            # Take actions w.r.t. the learning network
            actions = dq_networks.predict_values(next_states).argmax(dim=1).unsqueeze(1)
            # Predict values of those actions with target network
            q_values = dq_networks.predict_targets(next_states).gather(dim=1, index=actions)
            target_values[next_states_idx] += self.df * q_values
        return target_values.numpy()


class BaseTrainer(abc.ABC):
    """Defines the base class for a Trainer.

    A Trainer takes care of training the DQNetworks from the given target values.
    """

    @abc.abstractmethod
    def train(self, dq_networks, batch, targets, *args, **kwargs):
        """Perform one or several training steps on the DQNetworks object.

        Args:
            dq_networks (BaseDQNetworks): DQNetworks object to be trained. Must allow training
                with optimizer.
            batch (tuple): A tuple with all the data needed to train the network. The format of
                the content is left to the derived implementations.
            targets (tuple): Tuple with target values. The format of the content is left to the
                derived implementations.
        """
        pass


class DQNTrainer(BaseTrainer):

    def __init__(self, dq_networks, loss=None, optimizer=None, lr=1e-4, momentum=0.9):
        """Instantiate the DQNTrainer.

        Args:
            dq_networks (core.algorithms.dqnetworks.BaseDQNetworks): DQNetwork object to be trained.
                Must accept
            loss (function): A function that takes two Tensors (values and targets) and returns a
                scalar tensor with the loss value. If none, MSE loss is used.
            optimizer (torch.optim.Optimizer): Optimizer to use for training. If None,
                a SGD optimizer with the given learning rate and momentum will be used.
            lr (float): Learning rate for the SGD optimizer.
            momentum (float): Momentum for the SGD optimizer.
        """
        self.dq_networks = dq_networks
        self.loss = loss if loss is not None else F.mse_loss
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
            batch (tuple): A tuple (states, actions, rewards, next_states, next_states_idx) where:
                - states (ndarray): Numpy array with states on the rows.
                - actions (ndarray): Numpy array with actions correspondent to states on the rows.
                - rewards (ndarray): Numpy column vector with rewards correspondent to states and
                    actions on the rows
                - next_states (ndarray): Numpy array with next states on the rows. This may be
                    shorter than states, actions and rewards if the episode ended.
                - next_states_idx (ndarray): Numpy array containing indices to match next_states
                    with the appropriate elements of states. I.e. if the i-th element of
                    next_states_idx is k, then the k-th element of next_states refers to the k-th
                    element of states, actions, rewards.
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
