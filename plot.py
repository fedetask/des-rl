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
    parser.add_argument('--data', action='store', nargs='+', required=True)
    parser.add_argument('--max-plots', action='store', nargs=1, type=int, required=False,
                        default=[5])
    parser.add_argument('--contains', action='store', nargs='+', type=str, required=False,
                        default=[])
    parser.add_argument('--conf', action='store', nargs=1, type=float, required=False)
    args = parser.parse_args()

    plot_paths = []
    search_paths = args.data
    for path in search_paths:
        if os.path.isdir(path):
            for file in os.listdir(path):
                if len(args.contains) == 0 or any([s in file for s in args.contains]):
                    abs_path = os.path.join(path, file)
                    if os.path.isdir(abs_path):
                        search_paths.append(abs_path)
                    else:
                        plot_paths.append(abs_path)
        else:
            plot_paths.append(path)

    max_plots = args.max_plots[0]
    tot = 0
    while tot < len(plot_paths):
        start, end = tot, min(tot + max_plots, len(plot_paths))
        print(f'{start} {end}')
        for i in range(start, end):
            res = experiment_utils.read_result_numpy(
                os.path.dirname(plot_paths[i]), os.path.basename(plot_paths[i]))['train']
            x, y, var, _ = experiment_utils.merge_plots(res)
            stdev = np.sqrt(var)
            plt.plot(x, y, label=os.path.basename(plot_paths[i]))
            plt.fill_between(x, y - stdev, y+stdev, alpha=0.2)
            tot += 1
        plt.legend()
        plt.show()
        plt.close()
