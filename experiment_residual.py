import os

import copy
import gym
import torch
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt

import networks
import td3
import experiment_utils
import hardcoded_policies
from deepq import replay_buffers


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
                      exp_name_prefix='', exp_name_suffix=''):
    """This experiment pre-trains the actor network (and optionally the critic), then runs the TD3
    algorithm using the pre-trained actor (critic).

    Args:
        env (gym.Env): The gym environment.
        train_steps (int): Number of RL training steps.
        num_runs (int): Number of runs to perform. Each run is composed by pretraining and training.
        backbone_policy: The policy that will be used as a backbone.
        results_dir (str): Path to folder in which results must be stored.
        exp_name_prefix (str): Prefix to be added to experiment name.
        exp_name_suffix (str): Suffix to be added to experiment name.
    """
    exp_name = exp_name_prefix + backbone_policy.__name__ + exp_name_suffix
    if os.path.exists(os.path.join(results_dir, exp_name + '.npy')):
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
        td3_algorithm = td3.TD3(
            critic, actor, training_steps=train_steps, buffer_len=buffer_len,
            max_action=max_action, min_action=-max_action, critic_lr=critic_lr,
            actor_lr=actor_lr, df=df, evaluate_every=-1, backbone_policy=backbone_policy,
            epsilon_start=eps_start, epsilon_end=eps_end, batch_size=batch_size,
            epsilon_decay_schedule=eps_decay, checkpoint_every=checkpoint_every,
            checkpoint_dir='models/backbone/'
        )
        prefiller = replay_buffers.BufferPrefiller(
            num_transitions=buffer_prefill, collection_policy=backbone_policy,
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
                      results_dir, update_net_every=2, exp_name_prefix='', exp_name_suffix=''):
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
        actor, critic = get_actor_critic(state_len, action_len, max_action)

        # Train
        td3_algorithm = td3.TD3(critic, actor, training_steps=train_steps, max_action=max_action,
                                min_action=-max_action, critic_lr=critic_lr, actor_lr=actor_lr,
                                df=df, batch_size=batch_size, evaluate_every=-1,
                                epsilon_start=eps_start, epsilon_end=eps_end,
                                epsilon_decay_schedule=eps_decay,
                                update_targets_every=update_net_every,
                                checkpoint_every=checkpoint_every,
                                checkpoint_dir='models/standard/')
        prefiller = replay_buffers.BufferPrefiller(num_transitions=buffer_prefill)
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


def continue_training(env: gym.Env, train_steps, num_runs, actor_net, critic_net, buffer_len,
                      buffer_prefill, actor_lr, critic_lr, df, batch_size, eps_start, eps_end,
                      collection_policy_noise, collection_policy, eps_decay, checkpoint_every,
                      results_dir, exp_name_prefix='', exp_name_suffix=''):
    """Continue training the given networks and saves the results.

    Args:
        env (gym.Env): The gym environment.
        actor_net (torch.nn.Module): Actor network from which to resume training.
        critic_net (torch.nn.Module): Critic network from which to resume training.
        train_steps (int): Number of training steps.
        num_runs (int): Number of runs.
        results_dir (str): Path to folder in which results will be stored.
        exp_name_prefix (str): Prefix for the experiment name.
        exp_name_suffix (str): Suffix for experiment name.
    """
    params = locals()
    max_action = env.action_space.high[0]

    train_scores = []
    train_eval_scores = []
    for run in range(num_runs):
        actor, critic = copy.deepcopy(actor_net), copy.deepcopy(critic_net)

        # Train
        td3_algorithm = td3.TD3(critic, actor, training_steps=train_steps, max_action=max_action,
                                min_action=-max_action, critic_lr=critic_lr, batch_size=batch_size,
                                actor_lr=actor_lr, df=df, evaluate_every=-1,
                                epsilon_start=eps_start, epsilon_end=eps_end,
                                epsilon_decay_schedule=eps_decay,
                                checkpoint_every=checkpoint_every,
                                checkpoint_dir='models/continue/')
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
    exp_name = exp_name_prefix + continue_training.__name__ + exp_name_suffix
    experiment_utils.save_results_numpy(results_dir, exp_name, data_dict)


def backbone_experiment(env, train_steps, num_runs, actor_path, critic_path, exp_suffix,
                        buffer_len, buffer_prefill, actor_lr, critic_lr, df, batch_size, eps_start,
                        eps_end, eps_decay, collection_policy_noise, checkpoint_every):
    # Load actor and critic that we want to use
    actor = torch.load(actor_path)
    critic = torch.load(critic_path)[0]

    # Create backbone policy that uses the torch model
    def model_policy(state):
        with torch.no_grad():
            action = actor(
                torch.tensor(state).unsqueeze(0).float()
            )[0].detach().numpy()
        return action

    # Continue their training (makes a copy of actor and critic so they are not modified)
    continue_training(
        env=env, train_steps=train_steps, num_runs=num_runs, actor_net=actor, critic_net=critic,
        buffer_len=buffer_len, buffer_prefill=buffer_prefill, actor_lr=actor_lr,
        batch_size=batch_size, critic_lr=critic_lr, df=df, eps_start=eps_start, eps_end=eps_end,
        eps_decay=eps_decay, collection_policy_noise=eps_start,
        collection_policy=model_policy, checkpoint_every=checkpoint_every,
        results_dir=f'experiment_results/td3/continue/{env.unwrapped.spec.id}/',
        exp_name_suffix=exp_suffix,
    )

    # Train with backbone
    backbone_training(
        env=env, train_steps=train_steps, num_runs=num_runs, backbone_policy=model_policy,
        buffer_len=buffer_len, buffer_prefill=buffer_prefill, actor_lr=actor_lr, df=df,
        batch_size=batch_size, critic_lr=critic_lr, eps_start=eps_start, eps_end=eps_end,
        eps_decay=eps_decay,
        results_dir=f'experiment_results/td3/backbone/{env.unwrapped.spec.id}/',
        collection_policy_noise=collection_policy_noise, checkpoint_every=checkpoint_every,
        exp_name_suffix=exp_suffix
    )


if __name__ == '__main__':
    NUM_RUNS = 10
    TRAINING_STEPS = 15000
    BUFFER_PREFILL_STEPS = 2000
    BUFFER_LEN = TRAINING_STEPS + BUFFER_PREFILL_STEPS
    COLLECTION_POLICY_NOISE = 1.5
    CRITIC_LR = 1e-3
    ACTOR_LR = 1e-3
    UPDATE_NET_EVERY = 2
    DISCOUNT_FACTOR = 0.99
    BATCH_SIZE = 100
    EPSILON_START = 0.2
    EPSILON_END = 0.0
    EPSILON_DECAY_SCHEDULE = 'lin'
    CHECKPOINT_EVERY = 1000

    _env = gym.make('Pendulum-v0')

    """
    standard_training(
        env=_env, train_steps=TRAINING_STEPS, num_runs=NUM_RUNS,
        buffer_len=BUFFER_LEN, buffer_prefill=BUFFER_PREFILL_STEPS,
        actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, df=DISCOUNT_FACTOR, batch_size=BATCH_SIZE,
        eps_start=EPSILON_START, eps_end=EPSILON_END, eps_decay=EPSILON_DECAY_SCHEDULE,
        checkpoint_every=CHECKPOINT_EVERY,
        exp_name_suffix=f'_eps_{EPSILON_START}_to_{EPSILON_END}_prefill_{BUFFER_PREFILL_STEPS}',
        results_dir=f'experiment_results/td3/standard/{_env.unwrapped.spec.id}/'
    )
    """

    actor_path = 'models/standard/Pendulum-v0/actor_5000'
    critic_path = 'models/standard/Pendulum-v0/critic_5000'

    backbone_experiment(
        env=_env, train_steps=TRAINING_STEPS, num_runs=NUM_RUNS, actor_path=actor_path,
        critic_path=critic_path, buffer_len=BUFFER_LEN, buffer_prefill=BUFFER_PREFILL_STEPS,
        actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, df=DISCOUNT_FACTOR, batch_size=BATCH_SIZE,
        eps_start=EPSILON_START, eps_end=EPSILON_END, eps_decay=EPSILON_DECAY_SCHEDULE,
        collection_policy_noise=COLLECTION_POLICY_NOISE, checkpoint_every=CHECKPOINT_EVERY,
        exp_suffix=f'_actor_10000__eps_{EPSILON_START}_to_{EPSILON_END}_prefill_{BUFFER_PREFILL_STEPS}'
    )

