import os

import copy
import gym
import torch
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt

import common
from deepq import replay_buffers
import experiment_utils
import hardcoded_policies
import networks
import td3


def get_actor_critic(state_len, action_len, max_action):
    critic = networks.LinearNetwork(
        inputs=state_len + action_len,
        outputs=1,
        n_hidden_layers=2,
        n_hidden_units=256,
        activation=F.relu
    )
    actor = networks.LinearNetwork(
        inputs=state_len,
        outputs=action_len,
        n_hidden_layers=2,
        n_hidden_units=256,
        activation=F.relu,
        activation_last_layer=torch.tanh,
        output_weight=max_action
    )
    return actor, critic


def backbone_training(env: gym.Env, train_steps, num_runs, backbone_policy, buffer_len,
                      buffer_prefill, df, actor_lr, critic_lr, batch_size, eps_start, eps_end,
                      eps_decay, collection_policy_noise, checkpoint_every, results_dir,
                      exp_name_prefix='', exp_name_suffix='', checkpoint_subdir='/'):
    """This experiment pre-trains the actor network (and optionally the critic), then runs the TD3
    algorithm using the pre-trained actor (critic).

    Args:
        env (gym.Env): The gym environment.
        train_steps (int): Number of RL training steps.
        num_runs (int): Number of runs to perform. Each run is composed by pretraining and training.
        backbone_policy (Union[tuple, function]): The policy that will be used as a
            backbone. If tuple of strings, the first element is the actor path and the second
            is the critic path. The string can contain '%run' which will be replaced by the run
            number when loading the networks. Otherwise, it must be a tuple of functions for the
            actor and the critic respectively. In both cases, the critic may be None.
        results_dir (str): Path to folder in which results must be stored.
        exp_name_prefix (str): Prefix to be added to experiment name.
        exp_name_suffix (str): Suffix to be added to experiment name.
    """
    import inspect

    exp_name = exp_name_prefix + 'backbone_' + exp_name_suffix
    if os.path.exists(os.path.join(results_dir, exp_name + '.npy')):
        warn = 'Warning: ' + str(exp_name) + ' already exists. Skipping.'
        print(warn)
        return warn
    checkpoint_dir = os.path.join('models/backbone/', checkpoint_subdir)

    print('Starting experiment: ' + exp_name)

    state_len = env.observation_space.shape[0]
    action_len = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    train_scores = []
    for run in range(num_runs):
        print('Backbone training, run ' + str(run))

        if isinstance(backbone_policy[0], str):
            backbone_actor = torch.load(backbone_policy[0].replace('%run', f'{run}'))
            backbone_critic = None if backbone_policy[1] is None else torch.load(
                backbone_policy[1].replace('%run', f'{run}'))[0]
        elif inspect.isfunction(backbone_policy[0]):
            backbone_actor = backbone_policy[0]
            backbone_critic = None if backbone_policy[1] is None else backbone_policy[1]
        else:
            raise ValueError('Type for backbone_policy not understood.')

        actor, critic = get_actor_critic(state_len, action_len, max_action)

        # Train
        td3_algorithm = td3.TD3(
            critic, actor, training_steps=train_steps, buffer_len=buffer_len,
            max_action=max_action, min_action=-max_action, critic_lr=critic_lr,
            actor_lr=actor_lr, df=df, evaluate_every=-1, backbone_actor=backbone_actor,
            backbone_critic=backbone_critic, epsilon_start=eps_start, epsilon_end=eps_end,
            batch_size=batch_size, epsilon_decay_schedule=eps_decay,
            checkpoint_every=checkpoint_every, checkpoint_dir=checkpoint_dir
        )
        prefiller = replay_buffers.BufferPrefiller(
            num_transitions=buffer_prefill, collection_policy=backbone_actor,
            collection_policy_noise=collection_policy_noise, min_action=-max_action,
            max_action=max_action, use_residual=True
        )
        train_res = td3_algorithm.train(env, buffer_prefiller=prefiller)

        train_scores.append(experiment_utils.Plot(
            train_res['end_steps'], train_res['rewards'], name='train'))
    data_dict = {
        'train': train_scores
    }

    experiment_utils.save_results_numpy(results_dir, exp_name, data_dict)


def standard_training(env: gym.Env, train_steps, num_runs, buffer_len, buffer_prefill, actor_lr,
                      critic_lr, df, batch_size, eps_start, eps_end, eps_decay, checkpoint_every,
                      results_dir, update_net_every=2, exp_name_prefix='', exp_name_suffix='',
                      collection_policy=None, collection_policy_noise=None, checkpoint_subdir='/'):
    """Perform standard training with TD3. and saves the results.

    Args:
        env (gym.Env): The gym environment.
        train_steps (int): Number of training steps.
        num_runs (int): Number of runs.
        results_dir (str): Path to folder in which results will be stored.
        exp_name_prefix (str): Prefix for the experiment name.
        exp_name_suffix (str): Suffix for experiment name.
    """
    state_len = env.observation_space.shape[0]
    action_len = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    train_scores = []
    train_eval_scores = []
    for run in range(num_runs):
        checkpoint_dir = os.path.join('models/standard/', f'{checkpoint_subdir}_{run}')

        actor, critic = get_actor_critic(state_len, action_len, max_action)

        # Train
        td3_algorithm = td3.TD3(critic, actor, training_steps=train_steps, max_action=max_action,
                                min_action=-max_action, critic_lr=critic_lr, actor_lr=actor_lr,
                                df=df, batch_size=batch_size, evaluate_every=-1,
                                epsilon_start=eps_start, epsilon_end=eps_end,
                                epsilon_decay_schedule=eps_decay,
                                update_targets_every=update_net_every,
                                checkpoint_every=checkpoint_every,
                                checkpoint_dir=checkpoint_dir)
        prefiller = replay_buffers.BufferPrefiller(
            num_transitions=buffer_prefill, collection_policy=collection_policy,
            collection_policy_noise=collection_policy_noise, min_action=-max_action,
            max_action=max_action, use_residual=False
        )
        train_res = td3_algorithm.train(env, buffer_prefiller=prefiller)
        train_scores.append(experiment_utils.Plot(
            train_res['end_steps'], train_res['rewards'], name='train'))
        train_eval_scores.append(experiment_utils.Plot(
            train_res['eval_steps'], train_res['eval_scores'], name='eval'))
    data_dict = {
        'train': train_scores
    }
    exp_name = exp_name_prefix + standard_training.__name__ + exp_name_suffix
    experiment_utils.save_results_numpy(results_dir, exp_name, data_dict)


if __name__ == '__main__':
    NUM_RUNS = 15
    TRAINING_STEPS = 15000
    BUFFER_PREFILL = 2000
    BUFFER_LEN = 100000
    CRITIC_LR = 1e-3
    ACTOR_LR = 1e-3
    UPDATE_NET_EVERY = 2
    DISCOUNT_FACTOR = 0.99
    BATCH_SIZE = 100
    EPSILON_START = 0.2
    EPSILON_END = 0.0
    EPSILON_DECAY_SCHEDULE = 'lin'
    COLLECTION_POLICY_NOISE = 1
    CHECKPOINT_EVERY = 1

    _env = gym.make('Pendulum-v0')

    standard_training(
        env=_env, train_steps=TRAINING_STEPS * 2, num_runs=NUM_RUNS, buffer_len=BUFFER_LEN,
        buffer_prefill=BUFFER_PREFILL, actor_lr=ACTOR_LR,  critic_lr=CRITIC_LR,
        df=DISCOUNT_FACTOR, batch_size=BATCH_SIZE,  eps_start=EPSILON_START, eps_end=EPSILON_END,
        eps_decay=EPSILON_DECAY_SCHEDULE, checkpoint_every=CHECKPOINT_EVERY,
        update_net_every=UPDATE_NET_EVERY,
        results_dir=f'experiment_results/td3/standard/{_env.unwrapped.spec.id}',
        exp_name_suffix=f'_steps_{TRAINING_STEPS*2}_eps_{EPSILON_START}',
        checkpoint_subdir=f'steps_{TRAINING_STEPS*2}'
    )

    backbone_policy = (
        f'models/standard/steps_{TRAINING_STEPS*2}_%run/Pendulum-v0/actor_{TRAINING_STEPS}',
        f'models/standard/steps_{TRAINING_STEPS*2}_%run/Pendulum-v0/critic_{TRAINING_STEPS}')
    backbone_training(
        env=_env, train_steps=TRAINING_STEPS, num_runs=NUM_RUNS,
        backbone_policy=backbone_policy, buffer_len=BUFFER_LEN,
        buffer_prefill=BUFFER_PREFILL, df=DISCOUNT_FACTOR, actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR, batch_size=BATCH_SIZE, eps_start=EPSILON_START,
        eps_end=EPSILON_END, eps_decay=EPSILON_DECAY_SCHEDULE,
        collection_policy_noise=COLLECTION_POLICY_NOISE,
        checkpoint_every=CHECKPOINT_EVERY,
        results_dir='experiment_results/td3/backbone/backbone_continue/',
        checkpoint_subdir='backbone_continue'
    )