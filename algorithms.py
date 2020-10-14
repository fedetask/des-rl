"""This module contains some pre-cooked algorithms that you can use if do not care about building
yours from the available components.
"""

import abc
import torch
import torch.nn.functional as F
from torch import nn
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

    def __init__(self, critic_net, actor_net, training_steps=-1, max_action=None, min_action=None,
                 buffer_len=10000, prioritized_replay=True, df=0.99, batch_size=128,
                 critic_lr=0.001, actor_lr=0.001, train_actor_every=2, update_targets_every=2,
                 tau=0.005, target_noise=0.2, target_noise_clip=0.5, epsilon_start=0.8,
                 epsilon_end=0.05, epsilon_decay_schedule='exp', exploration_helper_policy=None,
                 exploration_helper_start_p=0.5, exploration_helper_end_p=0.05,
                 exploration_helper_schedule='lin', dtype=torch.float, evaluate_every=2500,
                 evaluation_episodes=5):
        self.critic_net = critic_net
        self.actor_net = actor_net
        self.training_steps = training_steps
        self.max_action = max_action
        self.min_action = min_action
        self.buffer_len = buffer_len
        self.prioritized_replay = prioritized_replay
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
        self.exploration_helper_policy = exploration_helper_policy
        self.exploration_helper_start_p = exploration_helper_start_p
        self.exploration_helper_end_p = exploration_helper_end_p
        self.exploration_helper_schedule = exploration_helper_schedule
        self.dtype = dtype
        self.evaluate_every = evaluate_every if evaluate_every > 0 else np.infty
        self.evaluation_episodes = evaluation_episodes

        self.cur_exploration_helper_p = exploration_helper_start_p
        if prioritized_replay:
            self.replay_buffer = replay_buffers.PrioritizedReplayBuffer(buffer_len)
        else:
            self.replay_buffer = replay_buffers.FIFOReplayBuffer(buffer_len)
        self.networks = deepqnetworks.DeepQActorCritic(
            critic_nets=[critic_net, copy.deepcopy(critic_net)],
            actor_net=actor_net,
            dtype=dtype
        )
        self.target_computer = computations.TD3TargetComputer(
            dqac_nets=self.networks,
            max_action=self.max_action,
            min_action=self.min_action,
            df=df,
            target_noise=target_noise,
            noise_clip=target_noise_clip,
            dtype=dtype
        )
        self.trainer = computations.TD3Trainer(
            self.networks,
            loss=torch.nn.MSELoss(),
            critic_lr=self.critic_lr,
            actor_lr=self.actor_lr,
            dtype=dtype
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
        start_steps = [0]
        end_steps = []
        predicted_target_values = []
        real_target_values = []
        episode_lengths = []
        eval_steps = []
        eval_avg_rewards = []
        critic_losses = []
        actor_losses = []

        if pretrain_steps > 0:
            prefiller = replay_buffers.UniformGymPrefiller()
            prefiller.fill(
                self.replay_buffer,
                env,
                num_transitions=pretrain_steps,
                prioritized_replay=self.prioritized_replay
            )

        next_eval = self.evaluate_every
        episode_rewards = []
        steps_range = tqdm.trange(self.training_steps, leave=True)
        state = env.reset()
        for step in steps_range:
            action = self.act(state, training=True, step=step)
            next_state, reward, done, info = env.step(action)
            step_res = self.step((state, action, reward, next_state, done), step)

            episode_rewards.append(reward)
            predicted_target_values.append(step_res['targets'])
            critic_losses.append(step_res['critic_loss'])
            actor_losses.append(step_res['actor_loss'])

            if done:
                rewards.append(np.sum(episode_rewards))
                end_steps.append(step)
                start_steps.append(step + 1)
                real_target_values.append(self._compute_real_targets(episode_rewards)[0])
                episode_lengths.append(len(episode_rewards))
                tqdm_descr = 'Episode reward: ' + str(round(rewards[-1], 3)) + ' length: '\
                             + str(episode_lengths[-1]) + ' step: ' + str(step)
                if len(eval_avg_rewards) > 0:
                    tqdm_descr += ' last evaluation: ' + str(eval_avg_rewards[-1])

                # Evaluate
                if step >= next_eval:
                    eval_avg_rewards.append(
                        self.evaluate(env, self.evaluation_episodes)
                    )
                    eval_steps.append(step)
                    next_eval += self.evaluate_every

                # Reset variables
                state = env.reset()
                episode_rewards = []

                # TDQM description
                steps_range.set_description(tqdm_descr)
            else:
                state = next_state
        return {
            'rewards': rewards,
            'end_steps': end_steps,
            'start_steps': start_steps,
            'predicted_targets': predicted_target_values,
            'real_targets': real_target_values,
            'eval_steps': eval_steps,
            'eval_avg_rewards': eval_avg_rewards,
            'critic_loss': critic_losses,
            'actor_loss': actor_losses
        }

    def step(self, transition, step_num):
        state, action, reward, next_state, done = transition
        next_state = next_state if not done else None
        transition = [state, action, reward, next_state]
        if self.prioritized_replay:
            transition.append(self.replay_buffer.avg_weight)
        self.replay_buffer.remember(transition)
        if len(self.replay_buffer.buffer) < self.batch_size:
            return
        if self.prioritized_replay:
            sampled_transitions, weights = self.replay_buffer.sample(self.batch_size)
        else:
            sampled_transitions = self.replay_buffer.sample(self.batch_size)
        batch = utils.split_replay_batch(sampled_transitions)
        targets = self.target_computer.compute_targets(batch)

        if self.prioritized_replay:
            self._update_prioritized_buffer(batch, sampled_transitions, targets)

        # Train critics at each step, and actor only every train_actor_every steps
        if self.prioritized_replay:
            train_res = self.trainer.train(
                batch,
                targets,
                train_actor=(step_num % self.train_actor_every == 0),
            )
        else:
            train_res = self.trainer.train(
                batch,
                targets,
                train_actor=(step_num % self.train_actor_every == 0),
                weights=weights
            )

        if step_num % self.update_targets_every == 0:
            self.networks.update_actor(mode='soft', tau=self.tau)
            self.networks.update_critic(mode='soft', tau=self.tau)

        return {
            'targets': np.mean(targets),
            'critic_loss': train_res['critic_loss'],
            'actor_loss': train_res['actor_loss'],
        }

    def evaluate(self, env, n_episodes, render=False):
        rewards = np.zeros(n_episodes)
        for e in range(n_episodes):
            state = env.reset()
            if render:
                env.render()
            done = False
            while not done:
                with torch.no_grad():
                    action = self.networks.actor_net(
                        torch.tensor(state, dtype=self.dtype).unsqueeze(0)
                    )[0]
                    action = action.numpy().clip(self.min_action, self.max_action)
                next_state, reward, done, _ = env.step(action)
                rewards[e] += reward
                state = next_state
        return rewards.mean()

    def act(self, state, training, step):
        action = None
        if training:
            if np.random.binomial(1, p=self.cur_exploration_helper_p):
                action = self.exploration_helper_policy(state)
                self.cur_exploration_helper_p -= \
                    (self.exploration_helper_start_p - self.exploration_helper_end_p) \
                    / self.training_steps
            else:
                with torch.no_grad():
                    action = self.networks.actor_net(
                        torch.tensor(state, dtype=self.dtype).unsqueeze(0)
                    )[0]
                    action += torch.randn_like(action) * self.epsilon_start
                    action = self.policy_train.act(action).numpy().clip(
                        self.min_action,
                        self.max_action
                    )
        return action

    def _td_error(self, states, actions, targets):
        with torch.no_grad():
            state_ts = torch.tensor(states, dtype=self.dtype)
            action_ts = torch.tensor(actions, dtype=self.dtype)
            q_value = self.networks.predict_values(state_ts, action_ts, mode='avg').numpy()
        return np.abs(targets - q_value).squeeze()

    def _update_prioritized_buffer(self, batch, transitions, targets):
        td_errors = self._td_error(batch[0], batch[1], targets)
        for i in range(len(transitions)):
            transitions[i][4] = td_errors[i]

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
    import hardcoded_policies

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

    td3 = TD3(
        critic_net=critic,
        actor_net=actor,
        min_action=min_action,
        max_action=max_action,
        training_steps=TRAINING_STEPS,
        buffer_len=1000000,
        prioritized_replay=True,
        df=0.99,
        batch_size=100,
        critic_lr=3e-4,
        actor_lr=3e-4,
        train_actor_every=2,
        update_targets_every=2,
        tau=0.005,
        target_noise=0.2,
        target_noise_clip=0.5,
        epsilon_start=0.1 * max_action,
        epsilon_end=0.05,
        epsilon_decay_schedule='lin',
        exploration_helper_policy=hardcoded_policies.mountain_car_deterministic,
        exploration_helper_start_p=0.7,
        exploration_helper_end_p=0.7,
        exploration_helper_schedule='const',
        dtype=torch.float,
        evaluate_every=2000,
        evaluation_episodes=10,
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

    plt.plot(train_result['eval_steps'], train_result['eval_avg_rewards'],
             label='Evaluation rewards')

    # Plotting actor loss
    actor_loss = train_result['actor_loss']
    for i in range(len(actor_loss)):
        j = 0
        while i + j < len(actor_loss) - 1 and actor_loss[i + j] is None:
            j += 1
        actor_loss[i] = actor_loss[i + j]
    actor_loss[-1] = actor_loss[-2] if actor_loss[-1] is None else actor_loss[-1]
    plt.plot(range(TRAINING_STEPS), actor_loss, label='Actor mean loss')

    plt.legend()

    plt.show()

    final_eval = td3.evaluate(env, 5)
    print('Final evaluation: ' + str(final_eval))
