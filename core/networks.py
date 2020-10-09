"""This module contains classes to fastly create neural networks.
"""
import torch
from torch import nn


class LinearNetwork(nn.Module):
    """A network composed of several linear layers.
    """

    def __init__(self, inputs, outputs, n_layers, n_units, activation=torch.tanh,
                 activation_last_layer=False):
        """Create a linear neural network with the given number of layers and units and
        the given activations.

        Args:
            inputs (int): Number of input nodes.
            outputs (int): Number of output nodes.
            n_layers (int): Total number of layers, including input and output layers.
            n_units (int): Number of units in the hidden layers.
            activation (function): The activation function to use. Use None for no activation.
            activation_last_layer (bool): Whether to apply the activation function in the
                output layer.
        """
        super().__init__()
        self.activation = activation
        self.activation_last_layer = activation_last_layer
        self.lin = nn.Linear(in_features=inputs, out_features=n_units)
        self.hidden_layers = [
            nn.Linear(in_features=n_units, out_features=n_units) for i in range(n_layers - 2)
        ]
        self.lout = nn.Linear(in_features=n_units, out_features=outputs)

    def forward(self, *inputs):
        """Forward pass on the concatenation of the given inputs.
        """
        cat_inputs = torch.cat([*inputs], 1)
        x = self.lin(cat_inputs)
        if self.activation is not None:
            x = self.activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
        x = self.lout(x)
        if self.activation is not None and self.activation_last_layer:
            x = self.activation(x)
        return x
