import os

import common
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


NUM_RUNS = 10
TRAINING_STEPS = 80000
BUFFER_PREFILL_STEPS = 10000
COLLECTION_POLICY_NOISE = 1.5
RL_CRITIC_LR = 0.5e-3
RL_ACTOR_LR = 0.5e-3
EPSILON_START = 0.2
EPSILON_END = 0.01
EPSILON_DECAY_SCHEDULE = 'lin'
CHECKPOINT_EVERY = 5000


def get_actor_critic(state_len, action_len, max_action):
    critic = networks.LinearNetwork(
        inputs=state_len + action_len,
        outputs=1,
        n_hidden_layers=3,
        n_hidden_units=128,
        activation=F.relu
    )
    actor = networks.LinearNetwork(
        inputs=state_len,
        outputs=action_len,
        n_hidden_layers=3,
        n_hidden_units=128,
        activation=F.relu,
        activation_last_layer=torch.tanh,
        output_weight=max_action
    )
    return actor, critic


def train_with_backbone(env: gym.Env, train_steps, num_runs, backbone_policy,
                        results_dir, exp_name_prefix=''):
    """This experiment pre-trains the actor network (and optionally the critic), then runs the TD3
    algorithm using the pre-trained actor (critic).

    Args:
        env (gym.Env): The gym environment.
        train_steps (int): Number of RL training steps.
        num_runs (int): Number of runs to perform. Each run is composed by pretraining and training.
        backbone_policy: The policy that will be used as a backbone.
        exp_name_prefix (str): Prefix to be added to experiment name.
        results_dir (str): Path to folder in which results must be stored.
    """
    exp_name = exp_name_prefix + backbone_policy.__name__
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
        td3_algorithm = td3.TD3(critic, actor, training_steps=train_steps, max_action=max_action,
                                min_action=-max_action, critic_lr=RL_CRITIC_LR,
                                actor_lr=RL_CRITIC_LR, evaluate_every=-1,
                                backbone_policy=backbone_policy,
                                epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                                epsilon_decay_schedule=EPSILON_DECAY_SCHEDULE,
                                checkpoint_every=CHECKPOINT_EVERY)
        prefiller = replay_buffers.BufferPrefiller(
            num_transitions=BUFFER_PREFILL_STEPS, collection_policy=backbone_policy,
            collection_policy_noise=COLLECTION_POLICY_NOISE, min_action=-max_action,
            max_action=max_action, use_residual=True)
        train_res = td3_algorithm.train(env, buffer_prefiller=prefiller)

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

    experiment_utils.save_results_numpy(results_dir, exp_name, data_dict)


def standard_training(env, train_steps, num_runs, results_dir, exp_name_prefix='',
                      exp_name_suffix=''):
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
                                min_action=-max_action, critic_lr=RL_CRITIC_LR,
                                actor_lr=RL_CRITIC_LR, evaluate_every=-1,
                                epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                                epsilon_decay_schedule=EPSILON_DECAY_SCHEDULE,
                                checkpoint_every=CHECKPOINT_EVERY)
        prefiller = replay_buffers.BufferPrefiller(num_transitions=BUFFER_PREFILL_STEPS)
        train_res = td3_algorithm.train(env, buffer_prefiller=prefiller)
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
    exp_name = exp_name_prefix + standard_training.__name__ + exp_name_suffix
    experiment_utils.save_results_numpy(results_dir, exp_name, data_dict)


if __name__ == '__main__':
    _env = gym.make('LunarLanderContinuous-v2')
    _results_dir = 'experiment_results/td3/lunar_lander'

    # Perform standard training
    standard_training(_env, train_steps=TRAINING_STEPS, num_runs=10, results_dir=_results_dir,
                      exp_name_suffix='_eps_0.2_0.01_lin')

    """
    # Perform training with backbone policy
    _actor = torch.load('models/LunarLanderContinuous-v2/actor_80000')
    
    def model_policy(state):
        with torch.no_grad():
            action = _actor(
                torch.tensor(state).unsqueeze(0).float()
            )[0].detach().numpy()
        return action

    train_with_backbone(
        exp_name_prefix='actor_80000', env=env, train_steps=TRAINING_STEPS, num_runs=10,
        backbone_policy=model_policy
    )
    """
