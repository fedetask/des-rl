"""This module provides implementations of the Q networks used in Deep Q Learning and all its
variants.

Q networks are generally two, a learning network and a target network, but different architectures
may be derived by extending the base class DQNetworks. The DQNetworks class defines the
functionalities that any implementation must provide in order to be compatible to the rest of the
package.
"""

import abc
import copy
import random
import numpy as np

import torch
from torch import nn


class BaseDQNetworks(abc.ABC):
    """The base class for a Deep Q Networks container.

    A Deep Q Network container is an object that stores the Q network(s) and provides methods for
    performing predictions and updates.
    """

    @abc.abstractmethod
    def predict_values(self, states, *args, **kwargs):
        """Predict the Q values for the given states.

        Args:
            states (numpy.ndarray): Numpy array containing the states for which to predict Q values.

        Returns:
            Tensor with the predicted Q values.
        """
        pass

    @abc.abstractmethod
    def predict_targets(self, states, *args, **kwargs):
        """Predict the target values for the given states.

        Args:
            states (numpy.ndarray): Numpy array containing the states for which to predict the
                target Q values.

        Returns:
            Tensor with the predicted target Q values.
        """
        pass

    @abc.abstractmethod
    def get_trainable_params(self, *args, **kwargs):
        """Get the trainable parameters of the model(s).

        Returns:
             Iterator[torch.nn.parameter.Parameter] to the parameters that an implementation of
             this class defines as trainable.
        """
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """General update function.

        In common DQN this corresponds to a target update, but other derived implementations may
        differ.
        """
        pass


class BaseDQActorCriticNetworks(abc.ABC):
    """The base class for Deep Q Actor Critic networks, used in many algorithms such as DDPG or TD3.
    """

    @abc.abstractmethod
    def predict_values(self, states, actions, *args, **kwargs):
        """Predict the Q values for the given state action pairs using the critic.

        Args:
            states (torch.Tensor): Tensor with the states.
            actions (torch.Tensor): Tensor with actions.

        Returns:
            Tensor with the predicted values for the given state-action pairs.
        """
        pass

    @abc.abstractmethod
    def predict_actions(self, states, *args, **kwargs):
        """Predict actions for the given states using the actor.

        Args:
            states (torch.Tensor): Tensor with states.

        Returns:
            Tensor with the predicted actions.
        """
        pass

    @abc.abstractmethod
    def predict_targets(self, states, actions, *args, **kwargs):
        """Predict the Q values for the given state action pairs using the target critic.

            Args:
                states (torch.Tensor): Tensor with the states.
                actions (torch.Tensor): Tensor with actions.

            Returns:
                Tensor with the predicted target values for the given state-action pairs.
        """
        pass

    @abc.abstractmethod
    def predict_target_actions(self, states, *args, **kwargs):
        """Predict actions for the given states using the target actor.

        Args:
            states (torch.Tensor): Tensor with states.

        Returns:
            Tensor with the predicted actions.
        """
        pass

    @abc.abstractmethod
    def get_trainable_params(self, *args, **kwargs):
        """Get the trainable parameters of the model(s).

        Returns:
             Tuple with parameters that an implementation of this class defines as trainable.
        """
        pass

    @abc.abstractmethod
    def update_actor(self, *args, **kwargs):
        """General actor update function.

        This may be implemented as hard update, Polyak update, etc.
        """
        pass

    @abc.abstractmethod
    def update_critic(self, *args, **kwargs):
        """General critic update function.

        This may be implemented as hard update, Polyak update, etc.
        """
        pass

    @abc.abstractmethod
    def type(self, dtype):
        """Set the type of all the network tensors to the given type.

        Args:
            dtype (torch.dtype): Desired type of the networks layers.
        """
        pass


# ------------------------------------ IMPLEMENTATIONS ---------------------------------------- #


class TargetDQNetworks(BaseDQNetworks):
    """Implementation of target Deep Q Networks, used in many Q learning algorithms such as DQN.
    """

    def __init__(self, q_network, target_network=None):
        """Save the given network and create the correspondent target network if not provided

        Args:
            q_network (nn.Module): Pytorch network used to compute Q values.
            target_network (nn.Module): Pytorch network used to compute target Q values. If None,
                a clone of q_network will be used.
        """
        self.q_network = q_network
        if target_network is not None:  # TODO re-initialize target weights?
            self.target_network = target_network
        else:
            self.target_network = copy.deepcopy(self.q_network)

    def predict_values(self, states, *args, **kwargs):
        return self.q_network(torch.tensor(states).float())

    def predict_targets(self, states, *args, **kwargs):
        return self.target_network(torch.tensor(states).float())

    def get_trainable_params(self, *args, **kwargs):
        return self.q_network.parameters()

    def update(self, mode='hard', tau=0.01, *args, **kwargs):
        """Perform hard Q network update, setting the target network weights with a copy of the
        learning network weights

        The update can be performed by copying parameters from the value network to the target
        network, or by Polyak averaging where target_new = (1 - tau) * target_net + tau * value_net.

        Args:
            mode (str): 'hard' for parameter copy, 'soft' for Polyak averaging.
            tau (float): Tau parameter of the soft (Polyak) update.
        """
        if mode == 'hard':
            self.target_network = copy.deepcopy(self.q_network)
        elif mode == 'soft':
            for param, target_param in zip(self.q_network.parameters(),
                                           self.target_network.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class DeepQActorCritic(BaseDQActorCriticNetworks):
    """This class implements a Deep Q Actor-Critic, used in many continuous control RL algorithms
    such as DDPG or TD3.

    A Deeq Q Actor-Critic is composed by one or many critic networks that predict the value of
    (s, a), and an actor network that computes the action for a given state. This implementation
    accepts a list of critic networks. Several options are given for computing values and
    targets. See predict_values() and predict_targets() for more details.

    Examples of Deep Q Actor-Critic architectures are DDPG, which uses only one critic network, or
    TD3, that uses two.
    """

    def __init__(self,
                 critic_nets,
                 actor_net,
                 critic_target_nets=None,
                 actor_target_net=None,
                 dtype=torch.float):
        """

        Args:
            critic_nets (list): A list of critic networks, that predict the value of (s, a).
            actor_net (torch.nn.Module): The actor network, that computes the action given a state.
            critic_target_nets (list): List of critic networks used to compute target values.
                If None, a copy the critic networks will be used.
            actor_target_net (torch.nn.Module): Actor network used to compute actions for
                targets. If None, a copy of actor_net will be used.
            dtype (torch.dtype): Type to which network layers will be set to. Use None for leave
                the networks unchanged.
        """
        super().__init__()
        self.critic_nets = critic_nets
        self.actor_net = actor_net
        if critic_target_nets is None:
            self.critic_target_nets = [copy.deepcopy(net) for net in self.critic_nets]
        else:
            self.critic_target_nets = critic_target_nets
        if actor_target_net is None:
            self.actor_target_net = copy.deepcopy(self.actor_net)
        else:
            self.actor_target_net = actor_target_net
        self.type(dtype)

    def predict_values(self, states, actions, mode='all', *args, **kwargs):
        """Predict the values for the given state-action pairs.

        Values can be computed from the critic networks in several modes:
            - 'first': Only the first critic network is used to compute the value.
            - 'min': Compute the value with all the networks and take the minimum.
            - 'avg': Compute the value with all the networks and take the average.
            - 'rand': Use a network chosen at random to predict the value.
            - 'all': Predicted values for all the network will be returned.

        Args:
            states (torch.Tensor): A Tensor with the states.
            actions (torch.Tensor): A Tensor with the actions.
            mode (str): How to compute the value.

        Returns:
            A Tensor with the values for the given state-action pairs. If mode 'all' is selected,
            a (N x k x 1) Tensor is returned, where N is the batch size and k is the number of
            networks.
        """
        assert states.size()[0] == actions.size()[0],\
            'Batch sizes for given states and actions do not correspond.'

        if mode == 'first':
            return self.critic_nets[0](states, actions)
        if mode == 'rand':
            net = random.choice(self.critic_nets)
            return net(states, actions)
        q_values = torch.stack([critic(states, actions) for critic in self.critic_nets], dim=1)
        if mode == 'avg':
            return torch.mean(q_values, dim=1)
        if mode == 'min':
            return torch.min(q_values, dim=1)
        if mode == 'all':
            return q_values

    def predict_actions(self, states, *args, **kwargs):
        actions = self.actor_net(states)
        return actions

    def predict_targets(self, states, actions, mode='all', grad=False, *args, **kwargs):
        """Predict the target values for the given state-action pairs.

        Targets can be computed from the critic networks in several modes:
            - 'first': Only the first critic network is used to compute the value.
            - 'min': Compute the value with all the networks and take the minimum.
            - 'avg': Compute the value with all the networks and take the average.
            - 'rand': Use a network chosen at random to predict the value.
            - 'all': Predicted target values for all the network will be returned.

        Args:
            states (torch.Tensor): A Tensor with the states.
            actions (torch.Tensor): A Tensor with the actions.
            mode (str): How to compute the value.
            grad (bool): Whether to track gradients for this computation.

        Returns:
            A Tensor with the target values for the given state-action pairs. If mode 'all' is
            selected, a (N x k x 1) Tensor is returned, where N is the batch size and k is
            the number of networks.
        """
        assert states.size()[0] == actions.size()[0], \
            'Batch sizes for given states and actions do not correspond.'

        if mode == 'first':
            if grad:
                return self.critic_target_nets[0](states, actions)
            else:
                with torch.no_grad():
                    return self.critic_target_nets[0](states, actions)
        if mode == 'rand':
            net = random.choice(self.critic_target_nets)
            if grad:
                return net(states, actions)
            else:
                with torch.no_grad():
                    return net(states, actions)

        if grad:
            q_values = torch.stack(
                [target_critic(states, actions) for target_critic in self.critic_target_nets],
                dim=1
            )
        else:
            with torch.no_grad():
                q_values = torch.stack(
                    [target_critic(states, actions) for target_critic in self.critic_target_nets],
                    dim=1
                )
        if mode == 'avg':
            return torch.mean(q_values, dim=1)
        if mode == 'min':
            return torch.min(q_values, dim=1)[0]
        if mode == 'all':
            return q_values

    def predict_target_actions(self, states, grad=False, *args, **kwargs):
        if grad:
            return self.actor_target_net(states)
        else:
            with torch.no_grad():
                return self.actor_target_net(states)

    def get_trainable_params(self, *args, **kwargs):
        """Return the critic and actor parameters.

        Returns:
            A tuple (critics_params, actor_params) where critics_params is a list of
            Iterator[torch.nn.parameter.Parameter] for each critic network, and actor params is an
            Iterator[torch.nn.parameter.Parameter] for the actor network.
        """
        critics_params = []
        for net in self.critic_nets:
            critics_params += list(net.parameters())
        return (critics_params, self.actor_net.parameters())

    def update_actor(self, mode='soft', tau=0.05, *args, **kwargs):
        """Update the target actor weights with the learning actor weights.

        The update can be performed by copying parameters from the value network to the target
        network, or by Polyak averaging where target_new = (1 - tau) * target_net + tau * value_net.

        Args:
            mode (str): 'hard' for parameter copy, 'soft' for Polyak averaging.
            tau (float): Tau parameter of the soft (Polyak) update.
        """
        if mode == 'hard':
            self.actor_target_net = copy.deepcopy(self.actor_net)
        elif mode == 'soft':
            for actor_param, actor_target_param in zip(
                    self.actor_net.parameters(),
                    self.actor_target_net.parameters()
            ):
                actor_target_param.data.copy_(
                    tau * actor_param.data + (1 - tau) * actor_target_param.data
                )

    def update_critic(self, mode='hard', tau=0.05, *args, **kwargs):
        """Update the target critic networks with the learning critic networks.

        The update can be performed by copying parameters from the learning critic networks to the
        target networks, or by Polyak averaging where target_new = (1 - tau) * target_net + tau *
        learning_net.

        Args:
            mode (str): 'hard' for parameter copy, 'soft' for Polyak averaging.
            tau (float): Tau parameter of the soft (Polyak) update.
        """
        if mode == 'hard':
            self.critic_target_nets = [copy.deepcopy(net) for net in self.critic_nets]
        elif mode == 'soft':
            for critic_net, critic_target_net in zip(self.critic_nets, self.critic_target_nets):
                for critic_param, critic_target_param in zip(
                        critic_net.parameters(),
                        critic_target_net.parameters()
                ):
                    critic_target_param.data.copy_(
                        tau * critic_param.data + (1 - tau) * critic_target_param.data
                    )

    def type(self, dtype):
        """Set all the networks to the given type.

        Args:
            dtype (torch.dtype): Desired type of the network layers.
        """
        for net in self.critic_nets:
            net.type(dtype)
        for net in self.critic_target_nets:
            net.type(dtype)
        self.actor_net.type(dtype)
        self.actor_target_net.type(dtype)
