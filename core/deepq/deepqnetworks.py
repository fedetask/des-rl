"""This module provides implementations of the Q networks used in Deep Q Learning and all its
variants.

Q networks are generally two, a learning network and a target network, but different architectures
may be derived by extending the base class DQNetworks. The DQNetworks class defines the
functionalities that any implementation must provide in order to be compatible to the rest of the
package.
"""
import abc
import copy

import torch
from torch import nn


class BaseDQNetworks(abc.ABC):
    """The base class of a Q networks implementation.
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


# ------------------------------------ IMPLEMENTATIONS ---------------------------------------- #


class TargetDQNetworks(BaseDQNetworks):
    """Implementation of Q network and target Q network used in the common DQN algorithm.
    """

    def __init__(self, q_network, target_network=None):
        """Save the given network and create the correspondent target network if not provided

        Args:
            q_network (nn.Module): Pytorch network used to compute Q values.
            target_network (nn.Module): Pytorch network used to compute target Q values. If None,
                a clone of q_network will be used.
        """
        self.q_network = q_network
        if target_network is not None:
            self.target_network = target_network
        else:
            self.target_network = copy.deepcopy(self.q_network)

    def predict_values(self, states, *args, **kwargs):
        return self.q_network(torch.tensor(states).float())

    def predict_targets(self, states, *args, **kwargs):
        return self.target_network(torch.tensor(states).float())

    def get_trainable_params(self, *args, **kwargs):
        return self.q_network.parameters()

    def update(self, *args, **kwargs):
        """Perform hard Q network update, setting the target network weights with a copy of the
        learning network weights
        """
        # TODO: which of the following is the best to use?
        # self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network = copy.deepcopy(self.q_network)
