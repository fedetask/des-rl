import abc
import copy

import torch
from torch import nn
from torch import optim
import numpy as np
import gym
import tqdm

import common
from deepq import deepqnetworks

class BasePretrainer(abc.ABC):

    @staticmethod
    def _collect_experience(env, collection_steps, collection_policy, df, dtype, *args, **kwargs):
        """Create a dataset from which to perform supervised learning.

        Args:
            env (gym.Env): The environment used to sample.
            collection_steps (int): Number of experience samples to collect.
            collection_policy: Function (numpy.ndarray) -> np.ndarray that maps a state to the
                corresponding action.
            df (float): Discount factor used to compute total returns.
            dtype (torch.dtype): Data type of the returned dataset.

        Returns:
            A tuple (states, actions, returns) where states is a (N, *state_shape) tensor with the
            sampled states, actions a (N, *action_shape) tensor with the correspondent actions, and
            returns a (N, 1) tensor with the total discounted return for each state-action pair.
        """
        states = np.empty((collection_steps, *env.observation_space.shape))
        actions = np.empty((collection_steps, *env.action_space.shape))
        rewards = np.empty((collection_steps, 1))
        next_states = np.empty((collection_steps, *env.observation_space.shape))
        dones = np.empty((collection_steps, 1))
        real_returns = np.empty((collection_steps, 1))
        episode_rewards = []
        state = env.reset()
        for step in range(collection_steps):
            action = collection_policy(torch.tensor(state).unsqueeze(0))[0].numpy()
            states[step] = state
            actions[step] = action
            next_state, r, done, info = env.step(action)
            rewards[step] = r
            next_states[step] = next_state
            dones[step] = float(done)
            episode_rewards.append(r)
            state = next_state
            if done:
                state = env.reset()
                total_returns = common.compute_real_targets(episode_rewards, df)
                total_returns = np.expand_dims(total_returns, 1)
                real_returns[step - total_returns.shape[0] + 1: step + 1] = total_returns
                episode_rewards = []
        states = torch.from_numpy(states).type(dtype)
        actions = torch.from_numpy(actions).type(dtype)
        rewards = torch.from_numpy(rewards).type(dtype)
        dones = torch.from_numpy(dones).type(dtype)
        next_states = torch.from_numpy(next_states).type(dtype)
        real_returns = torch.from_numpy(real_returns).type(dtype)
        return states, actions, rewards, next_states, dones, real_returns

    def sample_batch(self, batch_size, tensors):
        """Sample from the given tensors by sampling a list of indices and returning the
        correspondent elements of each tensor.

        Args:
            batch_size (int): Number of elements to sample.
            tensors (tuple): A tuple of tensors that must have the same first dimension.

        Returns:
            A tuple with the sampled elements for each dataset.

            Each element of the tuple is a tensor with the sampled elements from the
            corresponding dataset.
        """
        indices = np.arange(tensors[0].size()[0])
        sampled_idx = torch.from_numpy(np.random.choice(indices, batch_size))
        sampled = []
        for tensor in tensors:
            sampled.append(tensor[sampled_idx])
        return tuple(sampled)

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """Train the networks.
        """

    @staticmethod
    def _train_net(predictions, targets, loss_func, optimizer):
        loss = loss_func(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach().numpy()

    @staticmethod
    def eval(env, actor, eval_episodes, dtype):
        rewards = []
        for episode in range(eval_episodes):
            tot_reward = 0
            state = env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action = actor(
                        torch.from_numpy(state).unsqueeze(0).type(dtype))[0].detach().numpy()
                state, r, done, _ = env.step(action)
                tot_reward += r
            rewards.append(tot_reward)
        return np.array(rewards).mean()


class ActorCriticPretrainer(BasePretrainer):
    """Pretrainer for both actor and critic network.

    The actor is trained by supervised learning, sampling state-action pairs from the environment
    using an external collection policy. The critic is trained by supervised learning on the
    total discounted returns observed.
    """

    def __init__(self, env, actor, collection_policy, collection_steps, training_steps, critic=None,
                 actor_stop_steps=-1,  batch_size=128, actor_optimizer=None,
                 critic_optimizer=None, df=0.99, actor_lr=1e-3, critic_lr=0.5e-3,
                 actor_loss=torch.nn.MSELoss(), critic_loss=torch.nn.MSELoss(),
                 prevent_extrapolation=False, evaluate_every=500, eval_episodes=20,
                 dtype=torch.float):
        """Create the pretrainer for the actor critic networks.

        Args:
            env (gym.Env): The environment to sample from.
            actor (torch.nn.Module): Actor network to be trained.
            collection_policy: Function (numpy.ndarray) -> numpy.ndarray that maps states into
                actions, used to sample from the environment.
            collection_steps (int): Number of samples to take from the environment.
            training_steps (int): Number of training steps to perform. Each training step
                consists in sampling a batch and training critic and actor networks.
            critic (torch.nn.Module): Critic to be trained. Leave None for only actor training.
            actor_stop_steps (int): If > 0, the actor training will be stopped at actor_stop_steps.
            batch_size (int): Size of a sampled batch.
            actor_optimizer (torch.optim.Optimizer): Optimizer to train the actor. If None,
                Adam with the given actor_lr will be used.
            critic_optimizer (torch.optim.Optimizer): Optimizer to train the actor. If None,
                Adam with the given critic_lr rate will be used.
            actor_lr (float): Learning rate for actor optimizer.
            critic_lr (float):Learning rate for critic optimizer.
            actor_loss: Loss function to be used in actor supervised training.
            critic_loss: Loss function to be used in critic supervised training.
            evaluate_every (int): The agent will be evaluated every evaluate_every steps.
            eval_episodes (int): Number of episodes to run at each evaluation.
            dtype (torch.dtype): Type to use in tensor operations.
        """
        self.env = env
        self.actor = actor
        self.networks = deepqnetworks.DeepQActorCritic(
            critic_nets=[critic, copy.deepcopy(critic)],
            actor_net=actor,
        )
        self.collection_policy = collection_policy
        self.collection_steps = collection_steps
        self.training_steps = training_steps
        self.actor_stop_steps = actor_stop_steps
        self.batch_size = batch_size
        self.df = df
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.actor_loss = actor_loss
        self.critic_loss = critic_loss
        self.prevent_extrapolation = prevent_extrapolation
        self.evaluate_every = evaluate_every
        self.eval_episodes = eval_episodes
        self.dtype = dtype
        if self.actor_optimizer is None:
            self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        if self.critic_optimizer is None and critic is not None:
            self.critic_optimizer = torch.optim.Adam(
                self.networks.get_trainable_params()[0],
                lr=critic_lr
            )
        assert actor_stop_steps < 0 or critic is not None,\
            'Ambiguous setup: receiving actor_stop_steps > 0 when critic is None.'

    def train(self, model_save_dir=None, *args, **kwargs):
        """Perform the training.

        Returns:
            A dictionary with training statistics:
            {
                'actor_loss':    # Actor loss over time
                'critic_loss':   # Critic loss over time
                'eval_steps':    # Numpy array with steps at which the actor was evaluated.
                'eval_scores':   # Numpy array with evaluation scores (cumulative reward)
                                 # corresponding to eval_steps.
            }
        """
        states, actions, rewards, next_states, dones, tot_returns = self._collect_experience(
            self.env, df=self.df, collection_steps=self.collection_steps,
            collection_policy=self.collection_policy, dtype=self.dtype)
        actor_losses = []
        critic_losses = []
        eval_steps = []
        eval_scores = []
        training_range = tqdm.trange(self.training_steps, leave=True)
        for step in training_range:
            state_batch, action_batch, reward_batch, next_states_batch, dones_batch, return_batch \
                = self.sample_batch(
                    self.batch_size, (states, actions, rewards, next_states, dones, tot_returns)
                )
            if self.actor_stop_steps < 0 or step < self.actor_stop_steps:
                predicted_actions = self.actor(state_batch)
                actor_loss = self._train_net(
                    predicted_actions, action_batch, self.actor_loss, self.actor_optimizer)
                actor_losses.append(actor_loss)
            if self.networks is not None:
                predicted_values = self.networks.predict_values(
                    state_batch, action_batch, mode='first')
                targets = self._bootstrap_targets(reward_batch, next_states_batch, dones_batch)
                if self.prevent_extrapolation:
                    predicted_values_rand, targets_rand = self._generate_counter_samples(state_batch)
                    predicted_values = torch.cat([predicted_values, predicted_values_rand])
                    targets = torch.cat((targets, targets_rand))
                self.networks.update_critic(mode='soft', tau=0.05)
                critic_loss = self._train_net(
                    predicted_values, targets, self.critic_loss, self.critic_optimizer)
                training_range.set_description(
                    'Actor loss: ' + str(actor_loss) + ' critic loss: ' + str(critic_loss))
                critic_losses.append(critic_loss)

            if self.evaluate_every > 0 and step % self.evaluate_every == 0:
                eval_score = self.eval(self.env, self.actor, self.eval_episodes, self.dtype)
                eval_scores.append(eval_score)
                eval_steps.append(step)
        if self.evaluate_every > 0:
            eval_scores.append(self.eval(self.env, self.actor, self.eval_episodes, self.dtype))
            eval_steps.append(self.training_steps - 1)
        if model_save_dir is not None:
            common.save_models(
                models={'actor': self.networks.actor_net, 'critic': self.networks.critic_nets},
                dir=model_save_dir
            )
        return {
            'actor_loss': np.array(actor_losses),
            'critic_loss': np.array(critic_losses),
            'eval_steps': np.array(eval_steps),
            'eval_scores': np.array(eval_scores)
        }

    def _bootstrap_targets(self, rewards, next_states, dones):
        actions = self.collection_policy(next_states)
        with torch.no_grad():
            targets = self.networks.predict_targets(next_states, actions, mode='min')
        return rewards + dones * self.df * targets

    def _generate_counter_samples(self, states):
        """Generate prediction-target pairs to counter network extrapolation by sampling random
        actions and assigning them lower target value.

        Args:
            states (torch.Tensor):

        Returns:
           A tuple (predictions, targets) where predictions is a tensor with the critic
           predicted values for the given states and randomly sampled actions, and targets is a
           tensor with the corresponding penalized target value.
        """
        rand_actions = torch.stack(
            [torch.tensor(self.env.action_space.sample()) for i in range(states.shape[0])]
        )
        policy_actions = torch.stack(
            [torch.tensor(self.collection_policy(s)) for s in states]
        )
        predicted_values = self.networks.predict_values(states, rand_actions, mode='first')
        coef = (rand_actions - policy_actions)**2
        #coef = coef / (self.env.action_space.high - self.env.action_space.low)

        with torch.no_grad():
            policy_values = self.networks.predict_targets(states, policy_actions, mode='min')
            heuristic_values = policy_values - coef * torch.abs(policy_values)
        return predicted_values, heuristic_values


def plot_action_sample(env, critic, policy, n_states=5, n_steps=50):
    action_linspace = np.linspace(env.action_space.low, env.action_space.high, n_steps)
    states = [env.observation_space.sample() for i in range(n_states)]
    policy_actions = [policy(state) for state in states]

    for state, p_action in zip(states, policy_actions):
        # For each state compute critic value on each action value
        state_ts = torch.stack([torch.from_numpy(state) for i in range(n_steps)])
        actions_ts = torch.stack([torch.from_numpy(action) for action in action_linspace])
        with torch.no_grad():
            values = critic(state_ts, actions_ts).numpy()
            plt.plot(action_linspace, values, label=f'Policy action {p_action}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import networks
    import hardcoded_policies
    from matplotlib import pyplot as plt

    env = gym.make('Pendulum-v0')
    state_len = env.observation_space.shape[0]
    action_len = env.action_space.shape[0]
    actor_net = networks.LinearNetwork(
        inputs=state_len,
        outputs=action_len,
        n_hidden_layers=2,
        n_hidden_units=256,
        activation_last_layer=torch.tanh,
        output_weight=env.action_space.high[0],
        dtype=torch.float
    )

    critic_net = networks.LinearNetwork(
        inputs=state_len + action_len,
        outputs=1,
        n_hidden_layers=2,
        n_hidden_units=256,
    )

    training_steps = 5000
    pretrainer = ActorCriticPretrainer(
        env=env, actor=actor_net, critic=critic_net,
        collection_policy=hardcoded_policies.pendulum_torch,
        collection_steps=training_steps, training_steps=training_steps,
        actor_stop_steps=-1, dtype=torch.float, evaluate_every=-1, prevent_extrapolation=False,
        eval_episodes=0, actor_lr=1e-4, critic_lr=1e-4
    )
    res = pretrainer.train(model_save_dir='models/pretrain/')

    plot_action_sample(env, critic_net, hardcoded_policies.pendulum)

    eval_steps, eval_scores = res['eval_steps'], res['eval_scores']
    actor_loss, critic_loss = res['actor_loss'], res['critic_loss']
    policy_baseline = hardcoded_policies.eval_policy(hardcoded_policies.pendulum, env, 50)

    plt.plot(np.arange(actor_loss.shape[0]), actor_loss, label='Actor loss')
    plt.plot(np.arange(training_steps), critic_loss, label='Critic loss')
    plt.plot(eval_steps, eval_scores, label='Evaluation scores')
    plt.plot(np.arange(training_steps), np.ones(training_steps) * policy_baseline.mean(),
             label='Hardcoded policy score')
    plt.legend()
    plt.show()