import os

import common
import gym
import torch
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt

import networks
import algorithms
import experiment_utils
import hardcoded_policies
from deepq import replay_buffers

BACKBONE_RESULTS_DIR = 'experiment_results/td3/backbone_experiments/'

NUM_RUNS = 10
TRAINING_STEPS = 80000
BUFFER_PREFILL_STEPS = 30000
COLLECTION_POLICY_NOISE = 1.5
RL_CRITIC_LR = 0.5e-3
RL_ACTOR_LR = 0.5e-3
EPSILON_START = 0.1
EPSILON_END = 0.1
EPSILON_DECAY_SCHEDULE = 'const'


def get_actor_critic(state_len, action_len, max_action):
    critic = networks.LinearNetwork(
        inputs=state_len + action_len,
        outputs=1,
        n_hidden_layers=1,
        n_hidden_units=128,
        activation=F.relu
    )
    actor = networks.LinearNetwork(
        inputs=state_len,
        outputs=action_len,
        n_hidden_layers=1,
        n_hidden_units=128,
        activation=F.relu,
        activation_last_layer=torch.tanh,
        output_weight=max_action
    )
    return actor, critic


def train_with_backbone(env: gym.Env, train_steps, num_runs, backbone_policy, exp_name_prefix=''):
    """This experiment pre-trains the actor network (and optionally the critic), then runs the TD3
    algorithm using the pre-trained actor (critic).

    Args:
        env (gym.Env): The gym environment.
        train_steps (int): Number of RL training steps.
        num_runs (int): Number of runs to perform. Each run is composed by pretraining and training.
        backbone_policy: The policy that will be used as a backbone.
        exp_name_prefix (str): Prefix to be added to experiment name.
    """
    exp_name = exp_name_prefix + backbone_policy.__name__
    if os.path.exists(os.path.join(BACKBONE_RESULTS_DIR, exp_name + '.npy')):
        warn = 'Warning: ' + str(exp_name) + ' already exists. Skipping.'
        print(warn)
        return warn

    print('Starting experiment: ' + exp_name)

    state_len = env.observation_space.shape[0]
    action_len = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    train_scores = []
    for run in range(num_runs):
        print('Backbone training, run ' + str(run))

        actor, critic = get_actor_critic(state_len, action_len, max_action)

        # Train
        td3 = algorithms.TD3(critic, actor, training_steps=train_steps, max_action=max_action,
                             min_action=-max_action, critic_lr=RL_CRITIC_LR,
                             actor_lr=RL_CRITIC_LR, evaluate_every=-1,
                             backbone_policy=backbone_policy,
                             epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                             epsilon_decay_schedule=EPSILON_DECAY_SCHEDULE)
        prefiller = replay_buffers.BufferPrefiller(
            num_transitions=BUFFER_PREFILL_STEPS, collection_policy=backbone_policy,
            collection_policy_noise=COLLECTION_POLICY_NOISE, min_action=-max_action,
            max_action=max_action, use_residual=True)
        train_res = td3.train(env, buffer_prefiller=prefiller)

        train_scores.append(experiment_utils.Plot(
            train_res['end_steps'], train_res['rewards'], name='train'))
    data_dict = {
        'backbone_policy': backbone_policy.__name__,
        'buffer_prefill_steps': BUFFER_PREFILL_STEPS,
        'train_steps': train_steps,
        'rl_actor_lr': RL_ACTOR_LR,
        'rl_critic_lr': RL_CRITIC_LR,
        'train': train_scores
    }

    experiment_utils.save_results_numpy(BACKBONE_RESULTS_DIR, exp_name, data_dict)


def standard_training(env, train_steps, num_runs):
    """Perform standard training with TD3. and saves the results.

    Args:
        env (gym.Env): The gym environment.
        train_steps (int): Number of training steps.
        num_runs (int): Number of runs.
    """
    state_len = env.observation_space.shape[0]
    action_len = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    train_scores = []
    train_eval_scores = []
    for run in range(num_runs):
        actor, critic = get_actor_critic(state_len, action_len, max_action)

        td3 = algorithms.TD3(critic, actor, training_steps=train_steps, max_action=max_action,
                             min_action=-max_action, critic_lr=RL_CRITIC_LR,
                             actor_lr=RL_CRITIC_LR, evaluate_every=-1,
                             epsilon_decay_schedule=EPSILON_DECAY_SCHEDULE)
        prefiller = replay_buffers.BufferPrefiller(
            num_transitions=BUFFER_PREFILL_STEPS)
        train_res = td3.train(env, buffer_prefiller=prefiller)
        train_scores.append(experiment_utils.Plot(
            train_res['end_steps'], train_res['rewards'], name='train'))
        train_eval_scores.append(experiment_utils.Plot(
            train_res['eval_steps'], train_res['eval_scores'], name='eval'))
    data_dict = {
        'buffer_prefill_steps': BUFFER_PREFILL_STEPS,
        'train_steps': train_steps,
        'rl_actor_lr': RL_ACTOR_LR,
        'rl_critic_lr': RL_CRITIC_LR,
        'train': train_scores
    }
    exp_name = standard_training.__name__
    experiment_utils.save_results_numpy(BACKBONE_RESULTS_DIR, exp_name, data_dict)


def plot_experiment_1(dir, exp_results_filename, standard_training_filename, offset=True):
    exp_res = experiment_utils.read_result_numpy(dir, exp_results_filename)
    standard_train_res = experiment_utils.read_result_numpy(dir, standard_training_filename)
    x_pretrain, y_pretrain, _, _ = experiment_utils.merge_plots(exp_res['pretrain'])
    x_train, y_train, _, _ = experiment_utils.merge_plots(exp_res['train'])
    x_eval, y_eval, _, _, = experiment_utils.merge_plots(exp_res['eval'])
    if offset:
        x_train += x_pretrain[-1]
        x_eval += x_pretrain[-1]

    x_standard_train, y_standard_train, _, _ = experiment_utils.merge_plots(
        standard_train_res['train'])
    print(y_standard_train)
    x_standard_eval, y_standadrd_eval, _, _ = experiment_utils.merge_plots(
        standard_train_res['eval'])

    plt.plot(x_pretrain, y_pretrain, label='Pretraining evaluation scores')
    plt.plot(x_train, y_train, label='Training rewards')
    plt.plot(x_eval, y_eval, label='Training evaluation scores')
    plt.plot(x_standard_train, y_standard_train, label='Standard training rewards')
    plt.plot(x_standard_train, y_standard_train, label='Standard training evaluations')
    plt.legend()
    plt.show()


def plot(dir, max_in_plot=2, always_plot='standard_training.npy', cut_pretrain_at=150,
         only_files=None):
    if only_files is None:
        files = os.listdir(dir)
        files.remove(always_plot)
    else:
        files = only_files
    for i, file in enumerate(files):
        res = experiment_utils.read_result_numpy(dir, file)
        x, y, _, _ = experiment_utils.merge_plots(res['train'])
        if 'pretrain' in res:
            x_pretrain, y_pretrain, _, _ = experiment_utils.merge_plots(res['pretrain'])
            x_pretrain = x_pretrain[:cut_pretrain_at]
            y_pretrain = y_pretrain[:cut_pretrain_at]
            x += x_pretrain[-1]
            plt.plot(x_pretrain, y_pretrain, 'k')
        plt.plot(x, y, label=file)
        if (i + 1) % max_in_plot == 0 and i > 0 or i == len(files) - 1:
            res = experiment_utils.read_result_numpy(
                experiment_utils.PENDULUM_TD3_RESULTS_DIR, always_plot)
            x_alw, y_alw, _, _ = experiment_utils.merge_plots(res['train'])
            plt.plot(x_alw, y_alw, label=always_plot)
            plt.legend()
            plt.show()


if __name__ == '__main__':
    env = gym.make('LunarLanderContinuous-v2')
    _actor = torch.load('models/LunarLanderContinuous-v2/actor_80000')

    def model_policy(state):
        with torch.no_grad():
            action = _actor(
                torch.tensor(state).unsqueeze(0).float()
            )[0].detach().numpy()
        return action

    standard_training(env, train_steps=TRAINING_STEPS, num_runs=10)
    train_with_backbone(
        exp_name_prefix='actor_80000', env=env, train_steps=TRAINING_STEPS, num_runs=10,
        backbone_policy=model_policy
    )

    exit()
    standard = experiment_utils.read_result_numpy(experiment_utils.PENDULUM_TD3_RESULTS_DIR,
                                                  'standard_training.npy')
    std_x, std_y, _, _ = experiment_utils.merge_plots(standard['train'])

    backbone = experiment_utils.read_result_numpy(BACKBONE_RESULTS_DIR, 'backbone_policy.npy')
    backbone_x, backbone_y, _, _ = experiment_utils.merge_plots(backbone['train'])

    plt.plot(std_x, std_y, label='standard')
    plt.plot(backbone_x, backbone_y, label='backbone')
    plt.legend()
    plt.show()

