import json
import numpy as np
from scipy import interpolate
import os
import json
from matplotlib import pyplot as plt


EXPERIMENT_RESULT_DIR = 'experiment_results'


def merge_plots(plots, min_threshold=None, max_threshold=None):
    """Merge a list of plots that may have data points with different x values.

    The merging is performed by interpolating all the plots on common x values and averaging the
    result.

    Args:
        plots (list): List of numpy arrays where the first row corresponds to the x
            axis values and the second to the y axis values

    Returns:
        A tuple (x, y_avg, var, y_arrays) where:
            x (numpy.ndarray): x axis of the resulting averaged plot.
            y_avg (numpy.ndarray): Average of the interpolated plots.
            var (numpy.ndarray): Variance for each averaged point.
            y_arrays (list): List of the interpolations.
    """
    max_min = -np.infty
    min_max = np.infty
    # Taking a window of points from the max of the first x values to the min of the last values
    usable_plots = []
    skipped = 0
    for mat in plots:
        if min_threshold is not None and mat[0, 0] > min_threshold:
            skipped += 1
            continue
        if max_threshold is not None and mat[0, -1] < max_threshold:
            skipped += 1
            continue
        usable_plots.append(mat)
    if skipped > 0:
        print('Warning: skipped ' + str(skipped) + ' plots that do not respect x range limitations.')

    for mat in usable_plots:
        if mat[0, 0] > max_min:
            max_min = mat[0, 0]
        if mat[0, -1] < min_max:
            min_max = mat[0, -1]
    x_axis = np.arange(max_min, min_max, 10)
    y_arrays = [interpolate.interp1d(mat[0], mat[1])(x_axis) for mat in usable_plots]
    y_avg = np.sum(y_arrays, axis=0) / len(y_arrays)
    var = np.var(y_arrays, axis=0, ddof=1)
    return x_axis, y_avg, var, y_arrays


def plot_average_rewards(exp_dir, starts_with=None):
    """Plot the average rewards plot for each experiment in exp_dir

    Args:
        exp_dir (str): Path to the directory with experiment results.
        starts_with (str): Prefix of files to use. If None all files in exp_dir will be used.

    """
    files = [f for f in os.listdir(exp_dir) if starts_with is None or f.startswith(starts_with)]
    files.sort()
    files = [files[i] for i in range(len(files)) if i % 1 == 0]
    print(files)

    for file in files:
        filepath = exp_dir + '/' + file
        with open(filepath) as f:
            exp_results = json.load(f)
        n_steps = exp_results['parameters']['training_steps']
        p_hardcoded_start = round(exp_results['parameters']['p_hardcoded_start'], 1)
        p_hardcoded_end = round(exp_results['parameters']['p_hardcoded_end'], 1)

        rewards_list = exp_results['exp_rewards']
        end_steps_list = exp_results['exp_end_steps']

        plot_list = [
            np.array([end_steps, rewards])
            for end_steps, rewards in zip(end_steps_list, rewards_list)
        ]
        x, y_avg, var, y_arrays = merge_plots(plot_list)
        if p_hardcoded_start != p_hardcoded_end:
            label = 'p from ' + str(p_hardcoded_start) + ' to ' + str(p_hardcoded_end)
        else:
            label = 'p = ' + str(p_hardcoded_start)
        plt.plot(x, y_avg, label=label)
    plt.legend()
