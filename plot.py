import argparse
import os

from matplotlib import pyplot as plt
import numpy as np

import experiment_utils


def plot_comparison():
    RESULTS_DIR = 'experiment_results/td3/lunar_lander_backbone/'
    random_policy_reward, random_policy_std = -223, 129
    backbone_policy_reward, backbone_policy_std = 59.4, 271.6

    standard = experiment_utils.read_result_numpy(RESULTS_DIR,
                                                  'standard_lunar_lander_training.npy')
    std_x, std_y, std_var, _ = experiment_utils.merge_plots(standard['train'])

    backbone = experiment_utils.read_result_numpy(RESULTS_DIR, 'actor_lunar_lander_80000.npy')
    backbone_x, backbone_y, backbone_var, _ = experiment_utils.merge_plots(backbone['train'])

    plt.plot(std_x, std_y, 'b', label='standard')
    plt.fill_between(std_x, std_y - np.sqrt(std_var), std_y + np.sqrt(std_var),
                     color='b', alpha=0.2)
    plt.plot(backbone_x, backbone_y, 'g', label='backbone')
    plt.fill_between(backbone_x, backbone_y - np.sqrt(backbone_var),
                     backbone_y + np.sqrt(backbone_var),
                     color='g', alpha=0.2)

    plt.plot([0, backbone_x[-1]], [random_policy_reward] * 2, 'r', label='Random policy')

    plt.plot([0, backbone_x[-1]], [backbone_policy_reward] * 2, 'c', label='Backbone policy')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', nargs=1, required=True)
    args = parser.parse_args()

    res = experiment_utils.read_result_numpy(
        os.path.dirname(args.data[0]), os.path.basename(args.data[0]))['train']
    x, y, _, _ = experiment_utils.merge_plots(res)

    plt.plot(x, y)
    plt.show()
