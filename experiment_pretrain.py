"""This code runs the following experiments:

1. Standard training - the standard TD3 training

2. Pretrain actor - pretraining the actor from an hardcoded policy

3. Pretrain actor-critic - pretraining actor and critic from hardcoded policy and total returns

4. Pretrain actor-critic-delay - pretrain actor and critic, but delay the actor training

5. Prefill buffer with policy - use the hardcoded policy to prefill the replay buffer

"""

import os

import gym
import torch
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt

import networks
import pretrainers
import algorithms
import experiment_utils
import hardcoded_policies
import common
from deepq import replay_buffers

PRETRAIN_ACTOR_RL = 1e-3
PRETRAIN_CRITIC_RL = 1e-3
RL_CRITIC_LR = 0.5e-3
RL_ACTOR_LR = 0.5e-3


def get_actor_critic(state_len, action_len, max_action):
    critic = networks.LinearNetwork(
        inputs=state_len + action_len,
        outputs=1,
        n_hidden_layers=1,
        n_hidden_units=256,
        activation=F.relu
    )
    actor = networks.LinearNetwork(
        inputs=state_len,
        outputs=action_len,
        n_hidden_layers=1,
        n_hidden_units=256,
        activation=F.relu,
        activation_last_layer=torch.tanh,
        output_weight=max_action
    )
    return actor, critic


def train_from_pretrained_actor(env: gym.Env, collection_policy, pretrain_steps, train_steps,
                                buffer_prefill_steps, num_runs, buffer_prefill_from_policy=False,
                                train_critic=False, bootstrap_critic=True, actor_delay=0,
                                beta_start=1e-6, beta_schedule='lin', beta_steps=0,
                                prevent_extrapolation=True):
    """This experiment pre-trains the actor network (and optionally the critic), then runs the TD3
    algorithm using the pre-trained actor (critic).

    Args:
        env (gym.Env): The gym environment.
        collection_policy: Function (numpy.ndarray) -> numpy.ndarray that returns an action for
            a given state. Will be used to collect state-action pairs in the environment for
            training the actor.
        pretrain_steps (int): Number of supervised training steps when training the actor.
        train_steps (int): Number of RL training steps.
        buffer_prefill_steps: Number of steps to prefill the replay buffer with.
        num_runs (int): Number of runs to perform. Each run is composed by pretraining and training.
        buffer_prefill_from_policy (bool): Whether to use the given policy to prefill the buffer,
            or to sample random actions.
        train_critic (bool): Whether to train the critic.
        bootstrap_critic (bool): Whether to use bootstrapped estimates when pretraining the critic.
        actor_delay (int): Number of steps to wait before training the actor.
        beta_start (float): Initial value of beta.
        beta_schedule (str): Update schedule of beta.
        beta_steps (int): Number of steps in which beta is increased from beat_start to 1.
    """
    exp_name = collection_policy.__name__ + '_' + train_from_pretrained_actor.__name__
    if train_critic:
        exp_name += '_critic_bootstrap' if bootstrap_critic else '_critic_mc'
    if actor_delay > 0:
        exp_name += '_delay_actor_' + str(actor_delay)
    if buffer_prefill_from_policy:
        exp_name += '_prefill_from_policy'
    if beta_steps > 0:
        exp_name += '_beta_' + str(beta_schedule) + '_' + str(beta_start) + '_' + str(beta_steps)
    if prevent_extrapolation:
        exp_name += '_prevent_extrapolation'
    else:
        raise NotImplementedError('Not implemented no critic trick in pretrainer yet.')
    if os.path.exists(os.path.join(experiment_utils.PENDULUM_TD3_RESULTS_DIR, exp_name + '.npy')):
        warn = 'Warning: ' + str(exp_name) + ' already exists. Skipping.'
        print(warn)
        return warn

    print('Starting experiment: ' + exp_name)

    state_len = env.observation_space.shape[0]
    action_len = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    pretrain_scores = []
    train_scores = []
    train_eval_scores = []
    for run in range(num_runs):
        print('Pre-training, run ' + str(run))

        actor, critic = get_actor_critic(state_len, action_len, max_action)

        if beta_steps > 0:
            beta_updater = common.ParameterUpdater(
                start=beta_start, end=1., n_steps=beta_steps, update_schedule=beta_schedule)
        else:
            beta_updater = None

        # Pretrain actor and critic
        pretrainer = pretrainers.ActorCriticPretrainer(
            env=env, actor=actor, collection_policy=collection_policy,
            collection_steps=pretrain_steps, training_steps=pretrain_steps,
            critic=critic if train_critic else None, actor_lr=PRETRAIN_ACTOR_RL,
            critic_lr=PRETRAIN_CRITIC_RL, prevent_extrapolation=prevent_extrapolation,
            evaluate_every=200, eval_episodes=2
        )
        pretrain_res = pretrainer.train()

        # Train
        td3 = algorithms.TD3(critic, actor, training_steps=train_steps, max_action=max_action,
                             min_action=-max_action, critic_lr=RL_CRITIC_LR,
                             actor_lr=RL_CRITIC_LR, evaluate_every=-1,  # No eval
                             actor_start_train_at=actor_delay, actor_beta=beta_updater)
        train_res = td3.train(env, buffer_prefill_steps)

        pretrain_scores.append(experiment_utils.Plot(
            pretrain_res['eval_steps'], pretrain_res['eval_scores'], name='pretrain'))
        train_scores.append(experiment_utils.Plot(
            train_res['end_steps'], train_res['rewards'], name='train'))
        train_eval_scores.append(experiment_utils.Plot(
            train_res['eval_steps'], train_res['eval_scores'], name='eval'))
    data_dict = {
        'collection_policy': collection_policy.__name__,
        'pretrain_steps': pretrain_steps,
        'train_critic': train_critic,
        'pretrain_actor_lr': PRETRAIN_ACTOR_RL,
        'pretrain_critic_lr': PRETRAIN_CRITIC_RL,
        'buffer_prefill_steps': buffer_prefill_steps,
        'buffer_prefill_from_policy': buffer_prefill_from_policy,
        'train_steps': train_steps,
        'actor_delay': actor_delay,
        'rl_actor_lr': RL_ACTOR_LR,
        'rl_critic_lr': RL_CRITIC_LR,
        'pretrain': pretrain_scores,
        'train': train_scores
    }

    experiment_utils.save_results_numpy(
        experiment_utils.PENDULUM_TD3_RESULTS_DIR, exp_name, data_dict)


def standard_training(env, train_steps, buffer_prefill_steps, num_runs):
    """Perform standard training with TD3. and saves the results.

    Args:
        env (gym.Env): The gym environment.
        train_steps (int): Number of training steps.
        buffer_prefill_steps (int): Number of samples to fill the buffer with before training.
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
                             min_action=-max_action, critic_lr=RL_CRITIC_LR, actor_lr=RL_ACTOR_LR,
                             evaluate_every=-1)
        prefiller = replay_buffers.BufferPrefiller(num_transitions=buffer_prefill_steps)
        train_res = td3.train(env, prefiller)
        train_scores.append(experiment_utils.Plot(
            train_res['end_steps'], train_res['rewards'], name='train'))
        train_eval_scores.append(experiment_utils.Plot(
            train_res['eval_steps'], train_res['eval_scores'], name='eval'))
    data_dict = {
        'buffer_prefill_steps': buffer_prefill_steps,
        'train_steps': train_steps,
        'rl_actor_lr': RL_ACTOR_LR,
        'rl_critic_lr': RL_CRITIC_LR,
        'train': train_scores
    }
    exp_name = standard_training.__name__
    experiment_utils.save_results_numpy(
        experiment_utils.PENDULUM_TD3_RESULTS_DIR, exp_name, data_dict)


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


def run_experiments(train_standard=False):
    env = gym.make('Pendulum-v0')
    num_runs = 10
    buffer_prefill_steps = 5000  # Small to make policy actions count
    pretrain_steps = 5000
    train_steps = 15000

    if train_standard:
        standard_training(
            env,
            train_steps=pretrain_steps + train_steps,
            buffer_prefill_steps=buffer_prefill_steps,
            num_runs=num_runs
        )

    prefill_from_policy_vals = [True, False]
    train_critic_vals = [True, False]
    actor_delay_vals = [0, 2000, 5000]

    warnings = []
    for prefill_from_policy in prefill_from_policy_vals:
        for train_critic in train_critic_vals:
            for actor_delay in actor_delay_vals:
                warn = train_from_pretrained_actor(
                    env,
                    hardcoded_policies.pendulum,
                    pretrain_steps=pretrain_steps,
                    train_steps=train_steps,
                    buffer_prefill_steps=buffer_prefill_steps,
                    num_runs=num_runs,
                    buffer_prefill_from_policy=prefill_from_policy,
                    train_critic=train_critic,
                    actor_delay=actor_delay,
                    bootstrap_critic=True,
                )
                warnings.append(warn)
    for warn in warnings:
        print(warn)


def run_beta_experiments():
    env = gym.make('Pendulum-v0')
    num_runs = 10
    buffer_prefill_steps = 5000  # Small to make policy actions count
    pretrain_steps = 5000
    train_steps = 15000

    schedules = ['exp', 'lin']
    beta_steps = [250, 500, 1000]  # Actual steps are 2x since actor is trained every two
    actor_delays = [0, 1000, 1500]

    warnings = []
    for beta_step in beta_steps:
        for schedule in schedules:
            for actor_delay in actor_delays:
                warn = train_from_pretrained_actor(
                    env,
                    hardcoded_policies.pendulum,
                    pretrain_steps=pretrain_steps,
                    train_steps=train_steps,
                    buffer_prefill_steps=buffer_prefill_steps,
                    num_runs=num_runs,
                    buffer_prefill_from_policy=False,
                    train_critic=False,
                    actor_delay=actor_delay,
                    bootstrap_critic=True,
                    beta_schedule=schedule,
                    beta_steps=beta_step
                )
    for warn in warnings:
        print(warn)


def plot(max_in_plot=2, always_plot='standard_training.npy', cut_pretrain_at=150, only_files=None):
    if only_files is None:
        files = os.listdir(experiment_utils.PENDULUM_TD3_RESULTS_DIR)
        files.remove(always_plot)
    else:
        files = only_files
    for i, file in enumerate(files):
        res = experiment_utils.read_result_numpy(experiment_utils.PENDULUM_TD3_RESULTS_DIR, file)
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
            x, y, _, _ = experiment_utils.merge_plots(res['train'])
            plt.plot(x, y, label=always_plot)
            plt.legend()
            plt.show()


def test_adaptive_critic():
    env = gym.make('Pendulum-v0')
    num_runs = 10
    buffer_prefill_steps = 5000  # Small to make policy actions count
    pretrain_steps = 2000
    train_steps = 15000
    train_from_pretrained_actor(
        env,
        hardcoded_policies.pendulum,
        pretrain_steps=pretrain_steps,
        train_steps=train_steps,
        buffer_prefill_steps=buffer_prefill_steps,
        num_runs=num_runs,
        buffer_prefill_from_policy=False,
        train_critic=True,
        actor_delay=0,
        bootstrap_critic=True,
        beta_steps=-1,
        prevent_extrapolation=True
    )


if __name__ == '__main__':
    run_experiments(True)
    plot_only = os.listdir(experiment_utils.PENDULUM_TD3_RESULTS_DIR)
    plot_only = [p for p in plot_only if 'delay_actor_5000' in p]
    plot(max_in_plot=100, only_files=plot_only)