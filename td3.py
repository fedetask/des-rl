import copy
import numpy as np
import os
import torch
import torch.nn.functional as F
import tqdm

import common
from deepq import computations
from deepq import deepqnetworks
from deepq import policies
from deepq import replay_buffers

device = 'cpu'


class TD3:

    def __init__(self, critic_net, actor_net, training_steps, max_action=None, min_action=None,
                 backbone_actor=None, backbone_critic=None, buffer_len=100000,
                 prioritized_replay=False, df=0.99, batch_size=128, critic_lr=0.001,
                 actor_lr=0.001, actor_start_train_at=0, train_actor_every=2, actor_beta=None,
                 update_targets_every=2, tau=0.005, target_noise=0.2, target_noise_clip=0.5,
                 epsilon_start=0.15, epsilon_end=0.15, epsilon_decay_schedule='const',
                 decay_steps=-1, dtype=torch.float, evaluate_every=-1, evaluation_episodes=5,
                 checkpoint_every=-1, checkpoint_dir='models'):
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
            actor_beta (common.ParameterUpdater): A ParameterUpdater implementation that contains
                the value of beta, that is multiplied to the actor loss. The update() method is
                called at each actor training step. A increasing value of beta is useful to not
                destroy the actor policy when training from a pretrained actor.
            update_targets_every (int): Target nets will be updated once every
                update_targets_every steps with soft update.
            tau (float): Strength of soft update.
            target_noise (float): Stddev of the noise added to the target action.
            target_noise_clip (float): Maximum absolute value of target action added noise.
            epsilon_start (float): Initial stddev of exploration noise.
            epsilon_end (float): Final stddev of exploration noise.
            epsilon_decay_schedule (str): Exploration noise decay schedule 'const', 'lin', or 'exp'.
            dtype (torch.dtype): Type to use for all tensor computations.
            evaluate_every (int): The agent will be evaluated once every evaluate_every steps.
            evaluation_episodes (int): Number of episodes to run at each evaluation.
        """
        self.max_action = max_action
        self.min_action = min_action
        self.backbone_actor = backbone_actor
        self.backbone_critic = backbone_critic
        self.tau = tau  # Not private, one may want to play with it during training
        self._training_steps = training_steps
        self._prioritized_replay = prioritized_replay
        self._df = df
        self._batch_size = batch_size
        self._actor_start_train_at = actor_start_train_at
        self._train_actor_every = train_actor_every
        self._actor_beta = actor_beta
        self._update_targets_every = update_targets_every
        self._decay_steps = decay_steps
        self._dtype = dtype
        self._evaluate_every = evaluate_every
        self._evaluation_episodes = evaluation_episodes
        self._checkpoint_every = checkpoint_every
        self._checkpoint_dir = checkpoint_dir
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
            dtype=dtype,
            backbone_actor=backbone_actor,
            backbone_critic=backbone_critic
        )
        self.trainer = computations.TD3Trainer(
            self.networks,
            loss=torch.nn.MSELoss(),
            critic_lr=critic_lr,
            actor_lr=actor_lr,
            dtype=dtype,
            actor_beta=self._actor_beta,
            backbone_actor=backbone_actor,
            backbone_critic=backbone_critic
        )
        self.policy_train = policies.BaseEpsilonGaussianPolicy(
            start_epsilon=epsilon_start, end_epsilon=epsilon_end, decay_steps=training_steps,
            decay_schedule=epsilon_decay_schedule, min_action=min_action, max_action=max_action)

    def train(self, env, buffer_prefiller=None):
        """Perform TD3 training.

        Args:
            env (gym.Env): The Gym environment.
            buffer_prefiller (deepq.replay_buffers.BufferPrefiller): BufferPrefiller to use
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

        if buffer_prefiller is not None:
            buffer_prefiller.fill(self.replay_buffer, env)

        next_eval = self._evaluate_every
        episode_rewards = []
        steps_range = tqdm.trange(self._training_steps, leave=True)
        state = env.reset()
        for step in steps_range:
            if self.backbone_actor is not None:
                action, residual = self._act(state)
            else:
                action = self._act(state)
            next_state, reward, done, info = env.step(action)
            if self.backbone_actor is not None:
                step_res = self.step((state, residual, reward, next_state, done), step)
            else:
                step_res = self.step((state, action, reward, next_state, done), step)

            episode_rewards.append(reward)
            if step_res is None:
                continue
            predicted_target_values.append(step_res['targets'])
            critic_losses.append(step_res['critic_loss'])
            actor_losses.append(step_res['actor_loss'])

            if step % self._checkpoint_every == 0 and self._checkpoint_every > 0:
                common.save_models(
                    models={
                        f'actor_{step}': self.networks.actor_net,
                        f'critic_{step}': self.networks.critic_nets
                    },
                    dir=os.path.join(self._checkpoint_dir, env.unwrapped.spec.id)
                )

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
                if step >= next_eval and self._evaluate_every > 0:
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
        if self._checkpoint_every > 0:
            common.save_models(
                models={'actor': self.networks.actor_net, 'critic': self.networks.critic_nets},
                dir=os.path.join(self._checkpoint_dir, env.unwrapped.spec.id))
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
        sampled_transitions, info = self.replay_buffer.sample(self._batch_size)
        batch = common.split_replay_batch(sampled_transitions)
        targets = self.target_computer.compute_targets(batch)
        train_actor = step_num > self._actor_start_train_at \
            and step_num % self._train_actor_every == 0
        weights = None
        if self._prioritized_replay:
            self._update_prioritized_buffer(batch, sampled_transitions, targets)
            weights = info['weights']
        train_res = self.trainer.train(
            batch,
            targets,
            train_actor=train_actor,
            weights=weights,
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
                        torch.tensor(state, dtype=self._dtype, device=device).unsqueeze(0)
                    )[0]
                    action = action.numpy().clip(self.min_action, self.max_action)
                next_state, reward, done, _ = env.step(action)
                rewards[e] += reward
                state = next_state
        return float(rewards.mean())

    def _act(self, state):
        with torch.no_grad():
            net_action = self.networks.actor_net(
                torch.tensor(state, dtype=self._dtype, device=device).unsqueeze(0))[0]
        if self.backbone_actor is not None:
            with torch.no_grad():
                backbone_action = self.backbone_actor(
                    torch.tensor(state, dtype=self._dtype).unsqueeze(0))[0].detach().cpu().numpy()
            residual = self.policy_train.act(net_action).detach().cpu().numpy()
            action = (backbone_action + residual).clip(self.min_action, self.max_action)
            return action, residual
        else:
            action = self.policy_train.act(net_action).detach().cpu().numpy()
            action = action.clip(self.min_action, self.max_action)
            return action

    def _td_error(self, states, actions, targets):
        with torch.no_grad():
            state_ts = torch.tensor(states, dtype=self._dtype, device=device)
            action_ts = torch.tensor(actions, dtype=self._dtype, device=device)
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

    ENV_NAME = 'Pendulum-v0'
    TRAIN_STEPS = 15000
    PREFILL_STEPS = 2000
    SHOW_PLOT = True

    env = gym.make(ENV_NAME)
    action_len = env.action_space.shape[0]
    state_len = env.observation_space.shape[0]
    max_action = env.action_space.high[0]

    critic = networks.LinearNetwork(
        inputs=action_len+state_len,
        outputs=1,
        n_hidden_layers=1,
        n_hidden_units=16,
        activation=F.relu
    ).to(device)

    actor = networks.LinearNetwork(
        inputs=state_len,
        outputs=action_len,
        n_hidden_layers=1,
        n_hidden_units=16,
        activation=F.relu,
        activation_last_layer=torch.tanh,
        output_weight=max_action
    ).to(device)

    td3 = TD3(
        critic_net=critic,
        actor_net=actor,
        min_action=-max_action,
        max_action=max_action,
        training_steps=TRAIN_STEPS,
        df=0.99,
        batch_size=100,
        critic_lr=1e-3,
        actor_lr=1e-3,
        train_actor_every=2,
        update_targets_every=2,
        tau=0.005,
        target_noise=0.2,
        target_noise_clip=0.5,
        epsilon_start=0.2,
        epsilon_end=0.0,
        epsilon_decay_schedule='lin',
        dtype=torch.float,
        evaluate_every=-1,
        checkpoint_every=-1
    )
    prefiller = replay_buffers.BufferPrefiller(num_transitions=PREFILL_STEPS)
    train_result = td3.train(env)

    if SHOW_PLOT:
        plt.plot(train_result['end_steps'],
                 train_result['rewards'], label='Cumulative reward')

        start_steps = train_result['start_steps']
        if len(start_steps) != len(train_result['end_steps']):
            start_steps = start_steps[:-1]

        plt.plot(train_result['eval_steps'], train_result['eval_scores'],
                 label='Evaluation rewards')

        plt.legend()

        plt.show()

    final_eval = td3.evaluate(env, 5)
    print('Final evaluation: ' + str(final_eval))
