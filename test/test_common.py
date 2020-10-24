import pytest
import numpy as np

import common


def test_compute_targets():
    rewards = [1., 3., 5., 3., 4., 8.]
    df = 0.9
    expected = np.array([17.28532, 18.0948, 16.772, 13.08, 11.2, 8.])
    actual = np.array(common.compute_real_targets(rewards, df))
    assert np.array_equal(expected, actual)


def test_param_updater_const():
    nsteps = 50
    value = 0.1
    with pytest.raises(AssertionError):  # Correctly raise exception for start != end
        updater = common.ParameterUpdater(
            start=value, end=value + 0.1, n_steps=nsteps, update_schedule='const')
    updater = common.ParameterUpdater(
        start=value, end=value, n_steps=nsteps, update_schedule='const')
    for i in range(nsteps + 50):
        assert updater.cur_value == value
        updater.update()


def test_param_updater_lin_decreasing():
    nsteps = 50

    # Test decreasing
    start = 0.4
    end = -0.2
    updater = common.ParameterUpdater(start=start, end=end, n_steps=nsteps, update_schedule='lin')
    for i in range(nsteps):
        if i == 0:
            assert updater.cur_value == pytest.approx(start)
        else:
            assert start > updater.cur_value >= end
        updater.update()
    assert updater.cur_value == pytest.approx(updater.end)
    for i in range(nsteps):
        assert updater.cur_value == pytest.approx(updater.end)
        updater.update()

    # Test increasing
    start = -0.6
    end = 0.2
    updater = common.ParameterUpdater(start=start, end=end, n_steps=nsteps, update_schedule='lin')
    for i in range(nsteps):
        if i == 0:
            assert updater.cur_value == pytest.approx(start)
        else:
            assert start < updater.cur_value <= end
        updater.update()
    assert updater.cur_value == pytest.approx(updater.end)
    for i in range(nsteps):
        assert updater.cur_value == pytest.approx(updater.end)
        updater.update()


def test_parameter_updater_exp():
    nsteps = 50

    # Test decreasing
    start = 0.4
    end = 0.2
    updater = common.ParameterUpdater(start=start, end=end, n_steps=nsteps, update_schedule='exp')
    for i in range(nsteps):
        if i == 0:
            assert updater.cur_value == pytest.approx(start)
        else:
            assert start > updater.cur_value >= end
        updater.update()
    assert updater.cur_value == pytest.approx(updater.end)
    for i in range(nsteps):
        assert updater.cur_value == pytest.approx(updater.end)
        updater.update()

    # Test increasing
    start = 0.1
    end = 0.3
    updater = common.ParameterUpdater(start=start, end=end, n_steps=nsteps, update_schedule='exp')
    for i in range(nsteps):
        if i == 0:
            assert updater.cur_value == pytest.approx(start)
        else:
            assert start < updater.cur_value <= end
        updater.update()
    assert updater.cur_value == pytest.approx(updater.end)
    for i in range(nsteps):
        assert updater.cur_value == pytest.approx(updater.end)
        updater.update()