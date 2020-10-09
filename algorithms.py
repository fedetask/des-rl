"""This module contains some pre-cooked algorithms that you can use if do not care about building
yours from the available components.
"""

import abc
import torch
import torch.nn.functional as F
import numpy as np
import copy
import tqdm
import itertools

from core.deepq import deepqnetworks
from core.deepq import policies
from core.deepq import replay_buffers
from core.deepq import computations
from core.deepq import utils


class TD3:

    def __init__(self,
                 critic_net,
                 actor_net,
                 state_shape,
                 action_shape,
                 training_steps=-1,
                 max_action=None,
                 min_action=None,
                 buffer_len=10000,
                 df=0.99,
                 batch_size=128,
                 critic_lr=0.001,
                 actor_lr=0.001,
                 train_actor_every=2,
                 update_targets_every=2,
                 tau=0.005,
                 target_noise=0.2,
                 target_noise_clip=0.5,
                 epsilon_start=0.8,
                 epsilon_end=0.05,
                 epsilon_decay_schedule='exp'):
        self.critic_net = critic_net
        self.actor_net = actor_net
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.training_steps = training_steps
        self.max_action = max_action
        self.min_action = min_action
        self.df = df
        self.batch_size = batch_size
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.train_actor_every = train_actor_every
        self.update_targets_every = update_targets_every
        self.tau = tau
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_schedule = epsilon_decay_schedule
        self.epsilon_decay_steps = None

        self.replay_buffer = replay_buffers.FIFOReplayBuffer(buffer_len, state_shape, action_shape)
        self.networks = deepqnetworks.DeepQActorCritic(
            critic_nets=[critic_net, copy.deepcopy(critic_net)],
            actor_net=actor_net
        )
        self.target_computer = computations.TD3TargetComputer(
            dqac_nets=self.networks,
            max_action=self.max_action,
            min_action=self.min_action,
            df=df,
            target_noise=target_noise,
            noise_clip=target_noise_clip
        )
        self.trainer = computations.TD3Trainer(self.networks,
                                               loss=F.mse_loss,
                                               critic_lr=self.critic_lr,
                                               actor_lr=self.actor_lr
                                               )
        if epsilon_decay_schedule == 'const':
            self.policy_train = policies.FixedEpsilonGaussianPolicy(epsilon=epsilon_start)
        elif epsilon_decay_schedule == 'exp':
            self.policy_train = policies.ExponentialDecayEpsilonGaussianPolicy(
                start_epsilon=epsilon_start,
                end_epsilon=epsilon_end,
                decay_steps=self.training_steps
            )
        elif epsilon_decay_schedule == 'lin':
            self.policy_train = policies.LinearDecayEpsilonGaussianPolicy(
                start_epsilon=epsilon_start,
                end_epsilon=epsilon_end,
                decay_steps=self.training_steps
            )
        else:
            raise ValueError('The given epsilon_decay_schedule ' + str(epsilon_decay_schedule)
                             + ' is not understood.')

    def train(self, env, pretrain_steps=20000):
        rewards = []
        start_steps = []
        end_steps = []
        predicted_target_values = []
        real_target_values = []
        episode_lengths = []

        if pretrain_steps > 0:
            prefiller = replay_buffers.UniformGymPrefiller()
            prefiller.fill(self.replay_buffer, env, num_transitions=pretrain_steps)

        episode_rewards = []
        steps_range = tqdm.trange(self.training_steps, leave=True)
        state = env.reset()
        for step in steps_range:
            with torch.no_grad():
                action = self.networks.predict_actions(torch.Tensor(state).unsqueeze(0))[0]
                action = self.policy_train.act(action).numpy().clip(self.min_action, self.max_action)
            next_state, reward, done, info = env.step(action)
            step_res = self.step((state, action, reward, next_state, done), step)

            episode_rewards.append(reward)
            predicted_target_values.append(step_res['targets'])
            if len(episode_rewards) == 1:
                start_steps.append(step)

            if done:
                rewards.append(np.sum(episode_rewards))
                end_steps.append(step)
                real_target_values.append(self._compute_real_targets(episode_rewards)[0])
                episode_lengths.append(len(episode_rewards))

                # TDQM description
                steps_range.set_description('Episode reward: ' + str(round(rewards[-1], 3))
                                            + ' length: ' + str(episode_lengths[-1])
                                            + ' step: ' + str(step))

                # Reset variables
                state = env.reset()
                episode_rewards = []
            else:
                state = next_state
        return {
            'rewards': rewards,
            'end_steps': end_steps,
            'start_steps': start_steps,
            'predicted_targets': predicted_target_values,
            'real_targets': real_target_values
        }

    def step(self, transition, step_num):
        state, action, reward, next_state, done = transition
        next_state = next_state if not done else None
        self.replay_buffer.remember((state, action, reward, next_state))
        if len(self.replay_buffer.buffer) < self.batch_size:  # TODO: change when support len()
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = utils.split_replay_batch(transitions)
        targets = self.target_computer.compute_targets(batch)

        # Train critics at each step, and actor only every train_actor_every steps
        self.trainer.train(batch, targets, train_actor=(step_num % self.train_actor_every == 0))

        if step_num % self.update_targets_every == 0:
            self.networks.update_actor(mode='soft', tau=self.tau)
            self.networks.update_critic(mode='soft', tau=self.tau)

        return {
            'targets': np.mean(targets)
        }

    def _compute_real_targets(self, episode_rewards):
        targets = list(itertools.accumulate(
            episode_rewards[::-1],
            lambda tot, x: x + self.df * tot
        ))
        return targets[::-1]


if __name__ == '__main__':
    import gym
    from core import networks
    from matplotlib import pyplot as plt

    TRAINING_STEPS = 30000
    PRETRAIN_STEPS = 25000

    env = gym.make('MountainCarContinuous-v0')
    action_len = 1
    state_len = 2
    min_action = -1
    max_action = 1

    critic = networks.LinearNetwork(
        inputs=action_len+state_len,
        outputs=1,
        n_layers=1,
        n_units=256,
        activation=F.relu
    )

    actor = networks.LinearNetwork(
        inputs=state_len,
        outputs=action_len,
        n_layers=1,
        n_units=256,
        activation=F.relu
    )

    td3 = TD3(
        critic_net=critic,
        actor_net=actor,
        state_shape=(state_len,),
        action_shape=(action_len,),
        min_action=min_action,
        max_action=max_action,
        training_steps=TRAINING_STEPS,
        buffer_len=1000000,
        df=0.99,
        batch_size=128,
        critic_lr=3e-4,
        actor_lr=3e-4,
        train_actor_every=2,
        update_targets_every=2,
        tau=0.005,
        target_noise=0.2,
        target_noise_clip=0.5,
        epsilon_start=0.1 * max_action,
        epsilon_end=0.0,
        epsilon_decay_schedule='const'
    )

    train_result = td3.train(env, pretrain_steps=PRETRAIN_STEPS)

    plt.plot(train_result['end_steps'], train_result['rewards'], label='Cumulative reward')
    plt.plot(range(TRAINING_STEPS), train_result['predicted_targets'],
             label='Predicted target value')

    start_steps = train_result['start_steps']
    if len(start_steps) != len(train_result['end_steps']):
        start_steps = start_steps[:-1]

    plt.plot(start_steps, train_result['real_targets'],
             label='Real target value')
    plt.legend()

    plt.show()
