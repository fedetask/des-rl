import numbers

import torch
import numpy as np


def clamp(tensor, min_value, max_value):
    if isinstance(max_value, numbers.Number):
        clamped = tensor.clamp(min_value, max_value)
    elif isinstance(max_value, torch.Tensor):
        clamped = torch.max(
            torch.min(tensor, max_value),
            min_value
        )
    elif isinstance(max_value, np.ndarray):
        clamped = torch.max(
            torch.min(tensor, torch.from_numpy(max_value)),
            torch.from_numpy(min_value)
        )
    else:
        raise TypeError('The given max_action type ' + str(type(max_value)) +
                        ' is not understood.')
    return clamped
