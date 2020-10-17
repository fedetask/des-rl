import pytest
import numpy as np

import common


def test_compute_targets():
    rewards = [1., 3., 5., 3., 4., 8.]
    df = 0.9
    expected = np.array([17.28532, 18.0948, 16.772, 13.08, 11.2, 8.])
    actual = np.array(common.compute_real_targets(rewards, df))
    assert np.array_equal(expected, actual)
