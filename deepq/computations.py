"""The computations module provides target computers and trainers.

Target computers are objects that take care of computing the target values (e.g. in DQN) that are
generally in the form y = r + max_a' Q(s', a'), but can have different forms (e.g. in Double DQN or
in multi-step returns DQN).

Trainers take care of training the deep Q networks (provided in the deepqnetworks module) from
the given targets.
"""

import abc
import sys
import os

import torch
from torch.nn import functional as F
from torch.optim import lr_scheduler
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
import deepq.replay_buffers


class BaseTargetComputer(abc.ABC):
    """Base class for a target computer, that computes the target values for the given batch and
    DQNetworks.
    """

    def __init__(self, dtype=torch.float, *args, **kwargs):
        """Instantiate the target computer.

        Args:
            dtype (torch.dtype): Type to be used in computations.
        """
        self.dtype = dtype

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

    def __init__(self, dtype=torch.float, *args, **kwargs):
        """Instantiate the trainer.

        Args:
            dtype (torch.dtype): Type to be used in computations.
        """
        self.dtype = dtype

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

    def __init__(self, dq_networks, df=0.99, dtype=torch.dtype):
        """Instantiate the target computer.

        Args:
            dq_networks (core.deepq.deepqnetworks.BaseDQNetworks): The DQNetworks object that
                will be used to compute the targets.
            df (float): Discount factor in [0, 1].
            dtype (torch.dtype): Type to be used in computations.
        """
        super().__init__(dtype=dtype)
        self.dq_networks = dq_networks
        self.df = df

    def compute_targets(self, batch, *args, **kwargs):
        states, acts, rewards, next_states, next_states_idx = batch

        target_values = torch.tensor(rewards, dtype=self.dtype)  # Init target values with rewards
        with torch.no_grad():
            max_q = self.dq_networks.predict_targets(next_states).max(dim=1).values.unsqueeze(dim=1)
            target_values[next_states_idx] += self.df * max_q
        return target_values.numpy()


class DoubleQTargetComputer(BaseTargetComputer):
    """Implementation of the Double Q Network technique for computing the targets.
    """

    def __init__(self, dq_networks, df=0.99, dtype=torch.float):
        """Instantiate the double Q-learning target computer.

        Args:
            dq_networks (core.deepq.deepqnetworks.BaseDQNetworks): The DQNetworks object that
                will be used to compute the targets.
            df (float): The discount factor
            dtype (torch.dtype): Type to be used in computations.
        """
        super().__init__(dtype=dtype)
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
        _, _, rewards, next_states, next_states_idx = batch

        target_values = torch.tensor(rewards, dtype=self.dtype)  # Init target values with rewards

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
                 noise_clip=0.5,
                 dtype=torch.float,
                 backbone_actor=None,
                 backbone_critic=None):
        """Instantiate the TD3 target computer.

        Args:
            dqac_nets (core.deepq.deepqnetworks.DeepQActorCritic): The DeepQActorCritic containing
                the target critics and the actor that will be used to compute the target values.
            max_action (Union[float, numpy.ndarray, torch.Tensor]): Right-clipping value for target
                actions after adding the noise. If None, no clipping will be performed.
            min_action (Union[float, numpy.ndarray, torch.Tensor]): Right-clipping value for target
                actions after adding the noise. If None, no clipping will be performed.
            df (float): The discount factor.
            target_noise (Union[float, numpy.ndarray]): Std of the noise added to the target action
                along each dimension.
            noise_clip (Union[float, numpy.ndarray]): Noise will be clipped to
                (-noise_clip, +noise_clip).
            dtype (torch.dtype): Type to be used in computations.
        """
        super().__init__(dtype=dtype)
        assert type(max_action) == type(min_action), 'max_action and min_action must have same type'

        self.dqac_nets = dqac_nets
        self.max_action = max_action
        self.min_action = min_action
        self.df = df
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.backbone_actor = backbone_actor
        self.backbone_critic = backbone_critic
        if self.backbone_critic is not None:
            assert self.backbone_actor is not None, 'Backbone critic cannot be used without actor.'

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
        states, actions, rewards, next_states, next_states_idx = batch
        actions_ts = torch.tensor(actions, dtype=self.dtype)
        rewards_ts = torch.tensor(rewards, dtype=self.dtype)
        next_states_ts = torch.tensor(next_states, dtype=self.dtype)
        next_states_idx_ts = torch.tensor(next_states_idx, dtype=torch.long)

        with torch.no_grad():
            # Sampling and clamping noise
            noise_shape = (len(next_states_idx_ts), *actions_ts.size()[1:])
            noise = torch.randn(noise_shape) * self.target_noise
            noise = common.clamp(noise, -self.noise_clip, self.noise_clip)
            # Compute target actions and clamping if a max range is specified.
            if next_states_idx.shape[0] > 0:
                # Compute standard next actions and correspondent values
                net_action_ts = self.dqac_nets.predict_target_actions(next_states_ts) + noise

                # If a backbone critic and actor are given, compute residual targets
                if self.backbone_actor is not None:
                    with torch.no_grad():
                        backbone_actions = self.backbone_actor(next_states_ts)
                    net_action_ts = common.clamp(  # Clamp residual action such that the total
                        net_action_ts,  # action is within the valid range
                        self.min_action - backbone_actions,
                        self.max_action - backbone_actions
                    )
                else:
                    net_action_ts = torch.clamp(net_action_ts, self.min_action, self.max_action)

                target_values = self.dqac_nets.predict_targets(
                    next_states_ts,
                    net_action_ts,
                    mode='min'
                )
                if self.backbone_critic is not None:
                    with torch.no_grad():
                        target_values += self.backbone_critic(next_states_ts, backbone_actions)
                rewards_ts[next_states_idx_ts] += self.df * target_values
        return rewards_ts.detach().numpy()


class DQNTrainer(BaseTrainer):

    def __init__(
            self,
            dq_networks,
            optimizer=None,
            lr=1e-4,
            momentum=0.9,
            dtype=torch.float):
        """Instantiate the DQNTrainer.

        Args:
            dq_networks (core.algorithms.dqnetworks.BaseDQNetworks): DQNetwork object to be trained.
                Must accept
            optimizer (torch.optim.Optimizer): Optimizer to use for training. If None,
                a SGD optimizer with the given learning rate and momentum will be used.
            lr (float): Learning rate for the SGD optimizer.
            momentum (float): Momentum for the SGD optimizer.
            dtype (torch.dtype): Type to be used in computations.
        """
        super().__init__(dtype=dtype)
        self.dq_networks = dq_networks
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.SGD(
                dq_networks.get_trainable_params(),
                lr=lr,
                momentum=momentum
            )

    def train(self, batch, targets, importance_samples=None, *args, **kwargs):
        """Perform training steps on the given DQNetworks object for the given inputs and targets.

        Args:
            batch (tuple): A tuple (states, actions, rewards, next_states, next_states_idx) where
                each element is a numpy array. next_states does not contain next_states for
                states where the episode ended. For this reason, next_states_idx contains indices
                to match next_states with the appropriate elements of states. I.e. if the i-th
                element of next_states_idx is k, then the k-th element of next_states refers to
                the k-th element of states.
            targets (numpy.ndarray): Numpy column vector with targets on the rows.
            importance_samples (numpy.ndarray): Array with importance samples for each transition.
                In this case the loss is mean(importance_samples * (predictions, targets)**2).

        Returns:
            A tuple with training info: (loss, grads)
        """
        states, actions, _, _, next_states_idx = batch
        actions_idx_ts = torch.tensor(actions, dtype=torch.long)
        targets_ts = torch.tensor(targets, dtype=self.dtype)
        importance_samples_ts = torch.tensor(importance_samples)
        values = self.dq_networks.predict_values(states).gather(1, actions_idx_ts)
        loss = (values - targets_ts)**2
        if importance_samples is not None:
            loss *= importance_samples_ts
        loss = loss.mean()
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
                 actor_lr=0.5e-3,
                 actor_beta=None,
                 dtype=torch.float,
                 backbone_actor=None,
                 backbone_critic=None):
        """Instantiate the TD3 trainer.

        Args:
            dqac_networks (core.deepq.deepqnetworks.DeepQActorCritic): The DeepQActorCritic
                containing the networks to train.
            loss: A function that takes two Tensors (values and targets) and returns
                a scalar tensor with the loss value.
            critic_optimizer (torch.optim.Optimizer): Optimizer to use for training the critics.
                If None, an Adam optimizer with the given learning rate will be used.
            actor_optimizer (torch.optim.Optimizer): Optimizer to use for training the actor.
                If None, an Adam optimizer with the given learning rate will be used.
            critic_lr (float): Learning rate for the critic optimizer when using the default.
            actor_lr (float): Learning rate for the actor optimizer when using the default.
            actor_beta (common.ParameterUpdater): A parameter updater containing the value of
                beta, that is multiplied to the actor loss. The update() method is called every
                training step.
            dtype (torch.dtype): Type used for tensor computations.
        """
        super().__init__(dtype=dtype)
        self.dqac_networks = dqac_networks
        self.loss = loss
        self._actor_lr = actor_lr
        self.backbone_actor = backbone_actor
        self.backbone_critic = backbone_critic
        if critic_optimizer is None:
            self.critic_optimizer = torch.optim.Adam(
                dqac_networks.get_trainable_params()[0],
                lr=critic_lr
            )
        if actor_optimizer is None:
            self.actor_optimizer = torch.optim.Adam(
                dqac_networks.get_trainable_params()[1],
                lr=actor_lr
            )

    def train(self, batch, targets, train_actor=False, weights=None, *args, **kwargs):
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
            weights (numpy.ndarray): Optional array of weights for the gradient of each sample.
        """
        states, actions, _, _, _ = batch
        states_ts = torch.tensor(states, dtype=self.dtype)
        actions_ts = torch.tensor(actions, dtype=self.dtype)
        targets_ts = torch.tensor(targets, dtype=self.dtype)

        q_values = self.dqac_networks.predict_values(states_ts, actions_ts, mode='all')
        if self.backbone_critic is not None:
            with torch.no_grad():
                backbone_values = self.backbone_critic(states_ts, actions_ts)[:, None, :]
            q_values += backbone_values
        n_nets = q_values.shape[1]
        squared_errors = ((q_values - targets_ts[:, None, :]) ** 2).squeeze().sum(-1) / n_nets
        if weights is not None:
            squared_errors *= torch.tensor(weights, dtype=self.dtype)
        critic_loss = q_values.size()[1] * squared_errors.mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if train_actor:
            # Based on the assumption that Q(s, a) is separable wrt backbone critic and critic net
            act = self.dqac_networks.predict_actions(states_ts)
            actor_loss = - self.dqac_networks.predict_values(states_ts, act, 'rand').mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        return {
            'critic_loss': float(critic_loss.detach().numpy()),
            'actor_loss': float(actor_loss.detach().numpy()) if actor_loss is not None else None,
        }


class TD3OptionTraceTrainer:
    """Implementation of a TD3 trainer for a sub-policy.
    """

    def __init__(self,
                 dqac_networks,
                 df=0.99,
                 policy_noise=0.1,
                 rho_max=1,
                 lr=0.5e-3,
                 dtype=torch.float):
        """Instantiate the trainer.

        Args:
            dqac_networks (deepqnetworks.DeepQActorCritic): The DeepQActorCritic
                containing the networks to train.
            df (float): Discount factor.
            lr (float): Learning rate.
            dtype (torch.dtype): Type used for tensor computations.
        """
        super().__init__(dtype=dtype)
        self.dqac_networks = dqac_networks
        self.df = df
        self.policy_noise = policy_noise
        self.rho_max = rho_max
        self.optimizer = torch.optim.Adam(self.dqac_networks.get_trainable_params(), lr=lr)

    def train(self, batch, manager_critic, train_actor=False):
        """Perform one optimization step on the critic and actor networks for the given samples.

        Args:
            batch (list): List of Trajectory objects.
            train_actor (bool): Whether to train the actor network.
                importance_samples (numpy.ndarray): Numpy array with shape (N, 1) containing the
                importance samples to apply, with N being the number of state-action pairs in
                the batch.
        """
        n_episodes = len(batch)
        max_length = max(len(trajectory.rewards) for trajectory in batch)
        q_ret = None
        tot_loss = torch.tensor([0.])
        for t in range(1, max_length):
            mask = [i for i in range(n_episodes) if batch[i].length() - t >= 0]
            # old_policies contains log_p for each action
            states, actions, old_policies, rewards = self.extract(batch, t, mask)

            # Init retrace target if necessary and perform Bellman step
            if t == 1:
                q_ret = self._initial_q_ret(batch, manager_critic)
            q_ret[mask] = rewards + self.df * q_ret[mask]

            # Compute value and actor losses
            q_values = self.dqac_networks.predict_values(states, actions, mode='avg')
            value_loss = ((q_values - q_ret)**2 / 2).mean()
            actor_loss = torch.tensor([0.])
            cur_actions = self.dqac_networks.predict_actions(states)
            if train_actor:
                new_q_values = self.dqac_networks.predict_values(states, cur_actions)
                actor_loss = (-new_q_values).mean()
            tot_loss += value_loss + actor_loss

            # Compute importance weights
            distr = torch.distributions.normal.Normal(loc=cur_actions, scale=self.policy_noise)
            log_p_cur = distr.log_prob(actions)
            rho = torch.exp(log_p_cur - old_policies.log()).clamp(max=self.rho_max).detach()

            # Update retrace target. Here we use Q(states, pi(states)) as
            # a single sample-estimate of V(states)
            q_values_target = self.dqac_networks.predict_targets(states, actions, mode='min',
                                                                 grad=False)
            values_target = self._get_value(states, manager_critic)
            q_ret[mask] = rho * (q_ret[mask] - q_values_target) + values_target
        tot_loss.backward()
        self.optimizer.step()

    def extract(self, batch, t, mask):
        states = torch.cat(tuple(batch[i].states[-t] for i in mask), dim=0)
        actions = torch.cat(tuple(batch[i].actions[-t] for i in mask), dim=0)
        policies = torch.cat(tuple(batch[i].policies[-t] for i in mask), dim=0)
        rewards = torch.cat(tuple(batch[i].rewards[-t] for i in mask), dim=0)
        return states, actions, policies, rewards

    def _initial_q_ret(self, trajectories, manager_critic):
        # Compute the initial retrace action-value.
        q_rets = torch.empty(len(trajectories), 1)
        for i in range(len(trajectories)):
            if trajectories[i].last_state is None:
                q_rets[i, 0] = 0.
            else:
                with torch.no_grad():
                    # Compute off-policy value of the manager
                    q_values = manager_critic.predict_target_values(trajectories[i].last_state)
                    q_rets[i, 0] = q_values.max(dim=1).values
        return q_rets

    def _get_value(self, states, manager_critic):
        critic_q_values = manager_critic.predict_target_values(states)
        return critic_q_values.max(dim=1).values
