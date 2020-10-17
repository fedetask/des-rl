"""This module contains some pre-cooked algorithms that you can use if do not care about building
yours from the available components.
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy
import tqdm
import itertools

from deepq import computations, deepqnetworks, policies, replay_buffers
import common


class TD3:

    def __init__(self, critic_net, actor_net, training_steps=-1, max_action=None, min_action=None,
                 buffer_len=100000, prioritized_replay=False, df=0.99, batch_size=128,
                 critic_lr=0.001, actor_lr=0.001, actor_start_train_at=0, train_actor_every=2,
                 update_targets_every=2, tau=0.005, target_noise=0.2, target_noise_clip=0.5,
                 epsilon_start=0.15, epsilon_end=0.05, epsilon_decay_schedule='exp',
                 helper_policy=None, helper_start_p=0.5, helper_end_p=0.05,
                 helper_schedule='lin', dtype=torch.float, evaluate_every=2500,
                 evaluation_episodes=5):
        """Instantiate the TD3 algorithm.

        Args:
            critic_net (torch.nn.Module): Network to be used as critic.
            actor_net (torch.nn.Module): Network to be used as actor.
            training_steps (int): Number of steps in the training.
            max_action (Union[float], numpy.ndarray): Clipping value (high) for actions.
            min_action (Union[float], numpy.ndarray): Clipping value (low) for actions.
            buffer_len (int): Maximum length of the buffer.
            prioritized_replay (bool): Whether to use a prioritized replay buffer.
            df (float): Discount factor.
            batch_size (int): Number of transitions sampled in each training step.
            critic_lr (float): Learning rate for training the critic.
            actor_lr (float): Learning rate for training the actor.
            actor_start_train_at (int): Number of steps until which only the critic will be trained.
            train_actor_every (int): Actor net will be trained once every train_actor_every steps.
            update_targets_every (int): Target nets will be updated once every
                update_targets_every steps with soft update.
            tau (float): Strength of soft update.
            target_noise (float): Stddev of the noise added to the target action.
            target_noise_clip (float): Maximum absolute value of target action added noise.
            epsilon_start (float): Initial stddev of exploration noise.
            epsilon_end (float): Final stddev of exploration noise.
            epsilon_decay_schedule (str): Exploration noise decay schedule 'const', 'lin', or 'exp'.
            helper_policy: Function (numpy.ndarray) -> numpy.ndarray that returns an action
                for the given state.
            helper_start_p (float): Initial probability of taking an action with the helper policy.
            helper_end_p (float): Final probability of taking an action with the helper policy.
            dtype (torch.dtype): Type to use for all tensor computations.
            evaluate_every (int): The agent will be evaluated once every evaluate_every steps.
            evaluation_episodes (int): Number of episodes to run at each evaluation.
        """
        self.max_action = max_action
        self.min_action = min_action
        self.tau = tau  # Not private, one may want to play with it during training
        self.helper_policy = helper_policy  # Not private, one may want to change it
        self.helper_policy_updater = common.ParameterUpdater(  # Not private same reason
            helper_start_p, helper_end_p, training_steps, helper_schedule)
        self._training_steps = training_steps
        self._prioritized_replay = prioritized_replay
        self._df = df
        self._batch_size = batch_size
        self._actor_start_train_at = actor_start_train_at
        self._train_actor_every = train_actor_every
        self._update_targets_every = update_targets_every
        self._dtype = dtype
        self._evaluate_every = evaluate_every if evaluate_every > 0 else np.infty
        self._evaluation_episodes = evaluation_episodes

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
            critic_lr=critic_lr,
            actor_lr=actor_lr,
            dtype=dtype
        )
        self.policy_train = policies.BaseEpsilonGaussianPolicy(
            start_epsilon=epsilon_start, end_epsilon=epsilon_end, decay_steps=training_steps,
            decay_schedule=epsilon_decay_schedule, min_action=min_action, max_action=max_action)

    def train(self, env, buffer_prefill_steps=20000, buffer_collection_policy=None):
        """Perform TD3 training.

        Args:
            env (gym.Env): The Gym environment.
            buffer_prefill_steps (int): Number of transitions to add to replay buffer before
                starting the training.
            buffer_collection_policy: Function (numpy.ndarray) -> numpy.ndarray mapping states to
                actions. If not None, this function will be used to collect transitions.

        Returns:
            A dictionary with training statistics:
            {
                'rewards':              # Rewards per episode
                'end_steps':            # Time steps at which each episode ended
                'start_steps':          # Time steps at which each episode started
                'predicted_targets':    # Average predicted value at each time step
                'real_targets':         # Real value at each time step
                'eval_steps':           # Time steps at which the agent was evaluated
                'eval_scores':          # Evaluation scores
                'critic_loss':          # Loss of critic at each time step
                'actor_loss':           # Loss of actor at each time step
            }
        """
        rewards = []
        start_steps = [0]
        end_steps = []
        predicted_target_values = []
        real_target_values = []
        episode_lengths = []
        eval_steps = [0] if self._evaluate_every > 0 else []
        eval_avg_rewards = [self.evaluate(env, self._evaluation_episodes)]\
            if self._evaluate_every > 0 else []
        critic_losses = []
        actor_losses = []

        if buffer_prefill_steps > 0:
            prefiller = replay_buffers.BufferPrefiller()
            prefiller.fill(
                self.replay_buffer,
                env,
                num_transitions=buffer_prefill_steps,
                prioritized_replay=self._prioritized_replay,
                collection_policy=buffer_collection_policy
            )

        next_eval = self._evaluate_every
        episode_rewards = []
        steps_range = tqdm.trange(self._training_steps, leave=True)
        state = env.reset()
        for step in steps_range:
            action = self._act(state)
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
                real_target_values.append(
                    common.compute_real_targets(episode_rewards, self._df)[0])
                episode_lengths.append(len(episode_rewards))
                tqdm_descr = 'Episode reward: ' + str(round(rewards[-1], 3)) + ' length: '\
                             + str(episode_lengths[-1]) + ' step: ' + str(step)
                if len(eval_avg_rewards) > 0:
                    tqdm_descr += ' last evaluation: ' + \
                        str(eval_avg_rewards[-1])

                # Evaluate
                if step >= next_eval:
                    eval_avg_rewards.append(
                        self.evaluate(env, self._evaluation_episodes)
                    )
                    eval_steps.append(step)
                    next_eval += self._evaluate_every

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
            'eval_scores': eval_avg_rewards,
            'critic_loss': critic_losses,
            'actor_loss': actor_losses
        }

    def step(self, transition, step_num):
        state, action, reward, next_state, done = transition
        next_state = next_state if not done else None
        transition = [state, action, reward, next_state]
        if self._prioritized_replay:
            transition.append(self.replay_buffer.avg_td_error)
        self.replay_buffer.remember(transition)
        if len(self.replay_buffer.buffer) < self._batch_size:
            return

        # Sample a batch of transitions and train
        if self._prioritized_replay:
            sampled_transitions, weights = self.replay_buffer.sample(self._batch_size)
        else:
            sampled_transitions, _ = self.replay_buffer.sample(self._batch_size)
        batch = common.split_replay_batch(sampled_transitions)
        targets = self.target_computer.compute_targets(batch)
        train_actor = step_num > self._actor_start_train_at \
            and step_num % self._train_actor_every == 0
        if self._prioritized_replay:
            self._update_prioritized_buffer(batch, sampled_transitions, targets)
            train_res = self.trainer.train(
                batch,
                targets,
                train_actor=train_actor,
                weights=weights
            )
        else:
            train_res = self.trainer.train(
                batch,
                targets,
                train_actor=train_actor,
            )

        if step_num % self._update_targets_every == 0:
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
                        torch.tensor(state, dtype=self._dtype).unsqueeze(0)
                    )[0]
                    action = action.numpy().clip(self.min_action, self.max_action)
                next_state, reward, done, _ = env.step(action)
                rewards[e] += reward
                state = next_state
        return float(rewards.mean())

    def _act(self, state):
        if self.helper_policy is not None and np.random.binomial(1, p=self.cur_helper_p):
            action = self.helper_policy(state)
        else:
            with torch.no_grad():
                action = self.networks.actor_net(
                    torch.tensor(state, dtype=self._dtype).unsqueeze(0))[0]
                action = self.policy_train.act(action).numpy().clip(
                    self.min_action,
                    self.max_action
                )
        self.helper_policy_updater.update()
        return action

    def _td_error(self, states, actions, targets):
        with torch.no_grad():
            state_ts = torch.tensor(states, dtype=self._dtype)
            action_ts = torch.tensor(actions, dtype=self._dtype)
            q_value = self.networks.predict_values(
                state_ts, action_ts, mode='avg').numpy()
        return np.abs(targets - q_value).squeeze()

    def _update_prioritized_buffer(self, batch, transitions, targets):
        td_errors = self._td_error(batch[0], batch[1], targets)
        for i in range(len(transitions)):
            transitions[i][4] = td_errors[i]


if __name__ == '__main__':
    import gym
    import networks
    from matplotlib import pyplot as plt
    import hardcoded_policies

    TRAINING_STEPS = 30000
    PRETRAIN_STEPS = 25000

    env = gym.make('Pendulum-v0')
    action_len = 1
    state_len = 3
    min_action = -2
    max_action = 2

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
        prioritized_replay=False,
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
        helper_policy=hardcoded_policies.pendulum,
        helper_start_p=1.,
        helper_end_p=0.0,
        helper_schedule='const',
        dtype=torch.float,
        evaluate_every=2000,
        evaluation_episodes=10,
    )

    train_result = td3.train(env, buffer_prefill_steps=PRETRAIN_STEPS)

    plt.plot(train_result['end_steps'],
             train_result['rewards'], label='Cumulative reward')
    plt.plot(range(TRAINING_STEPS), train_result['predicted_targets'],
             label='Predicted target value')

    start_steps = train_result['start_steps']
    if len(start_steps) != len(train_result['end_steps']):
        start_steps = start_steps[:-1]

    plt.plot(start_steps, train_result['real_targets'],
             label='Real target value')

    plt.plot(train_result['eval_steps'], train_result['eval_scores'],
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
