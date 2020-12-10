import abc
import copy
import gym
import numpy as np
import torch
import tqdm

import common
from deepq import replay_buffers
from deepq import policies
from deepq import computations
from deepq import deepqnetworks
import networks
import hardcoded_policies


device = 'cpu'


class BaseSubPolicy(abc.ABC):

    @abc.abstractmethod
    def step(self, env, state, train):
        """Perform n_steps in the environment.

            Args:
                env (gym.Env): An environment in a valid state (not done).
                state (numpy.ndarray): The current state of the environment.
                train (bool): Whether to perform the training step.

            Returns:
                A tuple (next_state, rewards, *additional_info), where next_state is the new state
                after applying this policy for n_steps, and rewards is the list of obtained
                rewards. After rewards, any number of arguments can be passed.
        """
        pass

    @abc.abstractmethod
    def prefill(self, env):
        """If a learning policy, here is where the buffer is prefilled, otherwise just pass.
        """
        pass

    @abc.abstractmethod
    def get_name(self):
        """Return the name of the policy.
        """
        pass

    @abc.abstractmethod
    def train_step(self):
        pass

    @abc.abstractmethod
    def terminate(self, state):
        """Notify the sub-policy that its execution has been terminated.

        Args:
            state (numpy.ndarray): State at the time of termination.
        """
        pass

    @abc.abstractmethod
    def set_manager_critic(self, manager_critic):
        pass


class HardcodedSubPolicy(BaseSubPolicy):

    def __init__(self, policy, name='HardcodedSubPolicy', replay_buffer=None):
        """Create the hardcoded subpolicy.

        Args:
            policy: Function that maps a state (numpy.ndarray) into an action (numpy.ndarray).
            replay_buffer (replay_buffers.EpisodicReplayBuffer): A replay buffer to fill with
                transitions.
        """
        super().__init__()
        self.policy = policy
        self._name = name
        self.replay_buffer = replay_buffer

    def step(self, env, state, n_steps, train=False):
        """Perform n_steps in the environment.

        Args:
            env (gym.Env): An environment in a valid state (not done).
            state (numpy.ndarray): The current state of the environment.
            n_steps (int): Number of steps to perform in the environment.
            train (bool): Does nothing (for compatibility).

        Returns:
            A tuple (next_state, rewards), where next_state is the new state after applying this
            policy for n_steps, and rewards is the list of obtained rewards.
        """
        cur_state = state  # cur_state is always a numpy array
        states, actions, rewards, log_likelihoods = [], [], [], []
        for t in range(n_steps):
            # Take a step
            action = self.policy(cur_state)
            action_ts = torch.tensor(action).float().unsqueeze(0)
            log_likelihood = torch.zeros_like(action_ts)
            next_state, reward, done, info = env.step(action)
            self.replay_buffer.remember((common.state_to_tensor(cur_state), action_ts,
                                         log_likelihood, torch.tensor([[reward]]), done))

            # Store
            states.append(common.state_to_tensor(cur_state))
            actions.append(action)
            log_likelihoods.append(log_likelihood)

            cur_state = next_state if not done else None
            if done:
                break
        return states, actions, rewards, log_likelihoods, cur_state

    def prefill(self, env):
        pass

    def get_name(self):
        return self._name

    def train_step(self):
        pass

    def terminate(self, state):
        self.replay_buffer.cutoff(common.state_to_tensor(state))

    def set_manager_critic(self, manager_target_computer):
        pass


class TD3SubPolicy(BaseSubPolicy):

    def __init__(self, actor, critic, max_action, min_action, lr=0.5e-3,
                 df=0.99,  replay_buffer=None, buffer_len=100000, prefill_steps=10000,
                 batch_size=100, training=True, start_epsilon=0.15, end_epsilon=0.15,
                 epsilon_decay_schedule='const', update_every=2, tau=0.005, target_noise=0.05,
                 dtype=torch.float, name='TD3Subpolicy'):
        self._batch_size = batch_size
        self._training = training
        self._update_every = update_every
        self._tau = tau
        self._max_action = max_action
        self._min_action = min_action
        self._df = df
        self.target_noise = target_noise
        self._dtype = dtype
        self._name = name
        self.internal_step = 0
        self.networks = deepqnetworks.DeepQActorCritic(
            critic_nets=[critic, copy.deepcopy(critic)],
            actor_net=actor
        )
        if replay_buffer is None:
            self.replay_buffer = replay_buffers.EpisodicReplayBuffer(maxlen=buffer_len)
        else:
            self.replay_buffer = replay_buffer
        self.trainer = computations.TD3OptionTraceTrainer(
            dqac_networks=self.networks, df=0.99, policy_noise=0.1, lr=lr, rho_max=1.
        )
        self.policy_train = policies.BaseEpsilonGaussianPolicy(
            start_epsilon=start_epsilon, end_epsilon=end_epsilon, decay_steps=0,
            decay_schedule=epsilon_decay_schedule, min_action=-max_action, max_action=max_action
        )
        self._prefiller = replay_buffers.BufferPrefiller(num_transitions=prefill_steps)
        self.manager_critic = None

    def step(self, env, state, n_steps, train=True):
        """Perform n_steps in the environment.

        Args:
            env (gym.Env): An environment in a valid state (not done).
            state (numpy.ndarray): The current state of the environment.
            n_steps (int): Number of steps to perform in the environment.
            train (bool): Whether to perform the training step.

        Returns:
            A tuple (next_state, rewards), where next_state is the new state after applying this
            policy for n_steps, and rewards is the list of obtained rewards.
        """
        cur_state = state  # cur_state is always a numpy array
        states, actions, rewards, log_likelihoods = [], [], [], []
        for t in range(n_steps):
            # Take a step
            action, log_likelihood = self._act(cur_state)
            next_state, reward, done, info = env.step(common.tensor_to_action(action))
            self.replay_buffer.remember((common.state_to_tensor(cur_state), action, log_likelihood,
                                         torch.tensor([[reward]], done)))

            # Store
            states.append(common.state_to_tensor(cur_state))
            actions.append(action)
            log_likelihoods.append(log_likelihood)

            cur_state = next_state
        if train:
            self.train_step()
        return states, actions, rewards, log_likelihoods, cur_state

    def train_step(self):
        assert self.manager_critic is not None,\
            'TD3SubPolicy manager_critic is None. Did you forgot to set the manager critic?'
        if len(self.replay_buffer.buffer) < self._batch_size:
            return
        # Sample a batch of transitions and train
        sampled_transitions, info = self.replay_buffer.sample(self._batch_size)
        update = self.internal_step % self._update_every == 0
        self.trainer.train(sampled_transitions, self.manager_critic, train_actor=update)
        if update:
            self.networks.update_actor(mode='soft', tau=self._tau)
            self.networks.update_critic(mode='soft', tau=self._tau)
        self.internal_step += 1

    def _act(self, state):
        with torch.no_grad():
            net_action = self.networks.predict_actions(common.state_to_tensor(state))
        action = self.policy_train.act(net_action)
        log_likelihood = self._get_action_log_likelihood(
            actions=action,
            means=net_action,
            stds=self.policy_train.epsilon_updater.cur_value,
        )
        return action, log_likelihood

    @staticmethod
    def _get_action_log_likelihood(actions, means, stds):
        """Get the likelihood of the given actions for the given means and stds.

        Args:
            actions (torch.Tensor): (N, action_dim ) array with actions.
            means (torch.Tensor): (N, action_dim) array with the means of the action distribution.
            stds (torch.Tensor): Standard deviation of the action distribution.

        Returns:
            The computed log likelihood.
        """
        g = torch.distributions.normal.Normal(loc=means, scale=stds)
        return g.log_prob(actions)

    def prefill(self, env):
        #  self._prefiller.fill(self.replay_buffer, env)
        pass

    def get_name(self):
        return self._name

    def terminate(self, state):
        """Stores the current trajectory in the replay buffer. This function is called by the
        manager when it decides to swap sub-policy.
        """
        self.replay_buffer.cutoff(common.state_to_tensor(state))

    def set_manager_critic(self, manager_critic):
        self.manager_critic = manager_critic


class HierarchicalDeepQ:

    def __init__(self, q_net, sub_policies, training_steps, buffer_len=50000, prefill_steps=10000,
                 batch_size=128, df=0.99, lr=1e-3, update_every=10, dtype=torch.float,
                 train_subpolicy_steps=1, train_manager_steps=1, max_option_execution=20,
                 update_manager_target_every=10, epsilon_start=1.,
                 epsilon_end=0.01, decay_schedule='lin', train_interval=-1,
                 probs=None, alternate_training=False):
        """Create the Hierarchical Deep Q Learning algorithm.

        Args:
            sub_policies (list): List of BaseSubPolicy implementations.
            alternate_training (bool): Whether to alternate between training the manager olicy
                and the sub-policies for train_interval steps. If False, both are trained at
                each step.
            train_interval (int): If alternate_training is True, train_interval is the number of
                steps between switching from training the sub-policies to training the manager
                policy, and vice versa.
        """
        self._training_steps = training_steps
        self._sub_policies = sub_policies
        self._prefill_steps = prefill_steps
        self._batch_size = batch_size
        self._update_every = update_every
        self._training = False
        self._dtype = dtype
        self._train_interval = train_interval
        self._alternate_training = alternate_training
        self._train_subpolicy_steps = train_subpolicy_steps
        self._train_manager_steps = train_manager_steps
        self._max_option_execution = max_option_execution
        self._update_manager_target_every = update_manager_target_every
        self._replay_buffer = replay_buffers.FIFOReplayBuffer(maxlen=buffer_len)
        self._networks = deepqnetworks.TargetDQNetworks(q_network=q_net)
        self._target_computer = computations.DoubleQTargetComputer(
            dq_networks=self._networks, df=df
        )
        self._trainer = computations.DQNTrainer(dq_networks=self._networks, lr=lr)
        self._policy_train = policies.EpsilonGreedyPolicy(
            start_epsilon=epsilon_start, end_epsilon=epsilon_end, decay_steps=training_steps,
            decay_schedule=decay_schedule, probs=probs
        )
        for sub_policy in self._sub_policies:
            sub_policy.set_manager_critic(self.compute_values)

        if self._alternate_training:
            self._train_subpolicy, self._train_manager = True, False
        else:
            self._train_subpolicy, self._train_manager = True, True
        if self._alternate_training and self._train_interval <= 0:
            raise ValueError('Alternate training requires train interval > 0')

    def train(self, env):
        rewards = []
        sub_policy_rewards = [[] for _ in self._sub_policies]
        sub_policy_use = [[] for _ in self._sub_policies]
        predicted_values = [[] for _ in self._sub_policies]
        episode_end_steps = []
        # Prefill buffers for each sub-policy
        for policy in self._sub_policies:
            policy.prefill(env)
        self.prefill(env)  # Prefill options buffer

        episode_rewards = []
        episode_subpolicy_rewards = [0 for _ in self._sub_policies]
        episode_subpolicy_use = [0 for _ in self._sub_policies]
        episode_predicted_values = [[] for _ in self._sub_policies]
        episode_step = 0
        cur_option = None
        cur_option_step = 0
        state = env.reset()
        prog_bar = tqdm.trange(self._training_steps)
        for step in prog_bar:
            option, values = self._act(state)
            if cur_option is None:
                cur_option = option[0]
            # Notify the sub policy that we are switching option
            if option[0] != cur_option or cur_option_step > self._max_option_execution:
                self._sub_policies[option[0]].terminate()
            else:
                cur_option_step += 1
            cur_option = option[0]

            # Switch training policy if necessary
            if self._alternate_training:
                if step > 0 and (step % self._train_interval) == 0:
                    self._train_subpolicy = not self._train_subpolicy
                    self._train_manager = not self._train_manager
            # Perform step on chosen option (sub-policy)
            action, likelihood, step_reward, next_state = self._sub_policies[option[0]].step(
                env, state, self._train_subpolicy
            )
            done = next_state is None

            # Remember transition and perform training step if necessary
            transition = [state, option, action, likelihood, step_reward, next_state]
            self._replay_buffer.remember(transition)
            if self._train_manager:
                self.train_step(step)
            state = next_state

            # Update statistics
            episode_rewards.append(step_reward)
            episode_subpolicy_rewards[option[0]] += step_reward
            episode_subpolicy_use[option[0]] += 1
            for i in range(len(self._sub_policies)):
                episode_predicted_values[i].append(values[i])
            episode_step += 1

            if done:
                state = env.reset()
                rewards.append(sum(episode_rewards))
                for i in range(len(episode_subpolicy_use)):
                    sub_policy_use[i].append(episode_subpolicy_use[i] / episode_step)
                    sub_policy_rewards[i].append(
                        episode_subpolicy_rewards[i] / (episode_subpolicy_use[i] + 1e-8)
                    )
                    predicted_values[i].append(np.mean(episode_predicted_values[i]))
                episode_subpolicy_use = [0 for _ in self._sub_policies]
                episode_subpolicy_rewards = [0 for _ in self._sub_policies]
                episode_predicted_values = [[] for _ in self._sub_policies]
                episode_rewards = []
                episode_step = 0
                episode_end_steps.append(step)
                descr_str = f'Episode reward: {rewards[-1]} '
                for i in range(len(self._sub_policies)):
                    descr_str += f' {self._sub_policies[i].get_name()}:' \
                                 f' use: {round(sub_policy_use[i][-1], 3)}' \
                                 f' rew: {round(sub_policy_rewards[i][-1], 3)}'
                prog_bar.set_description(descr_str)

        return {
            'end_steps': episode_end_steps,
            'rewards': rewards,
            'subpolicy_use': sub_policy_use,
            'subpolicy_rewards': sub_policy_rewards,
            'predicted_values': predicted_values
        }

    def train_step(self, step):
        """Perform a traninig step on the manager network.

        Args:
            step (int): The current training step.
        """
        if len(self._replay_buffer.buffer) < self._batch_size:
            return
        # Sample a batch of transitions and train
        sampled_transitions, info = self._replay_buffer.sample(self._batch_size)
        batch = common.split_option_replay_batch(sampled_transitions)
        states, options, actions, likelihoods, rewards, next_states, next_states_idx = batch
        importance_samples = self._compute_importance_samples(states, options, actions, likelihoods)
        batch = states, options, rewards, next_states, next_states_idx
        targets = self._target_computer.compute_targets(batch)
        self._trainer.train(batch=batch, targets=targets, importance_samples=importance_samples)
        if (step % self._update_manager_target_every) == 0 and step > 0:
            self._networks.update(mode='hard')

    def _compute_importance_samples(self, states, options, actions, likelihoods, retrace=True):
        # Grouping states, actions, likelihoods by sub-policy based on options
        n = states.shape[0]
        states_ts = torch.tensor(states)
        actions_ts = torch.tensor(actions)
        option_ids = [[] for i in range(len(self._sub_policies))]  # One array per sub-policy
        target_likelihoods = np.empty(n)
        # Assign indices of state-action pairs to the correspondent option
        for i, o in enumerate(options):
            option_ids[o[0]].append(i)
        option_ids = [torch.tensor(ids, dtype=torch.long) for ids in option_ids]

        # For each sub-policy take correspondent states and actions and compute log-likelihood
        for i, sub_p in enumerate(self._sub_policies):
            ids = option_ids[i]
            target_likelihoods[ids] = sub_p.get_target_likelihoods(states_ts[ids], actions_ts[ids])
        importance_samples = np.exp(target_likelihoods - likelihoods)
        if retrace:
            importance_samples = np.minimum(importance_samples, 1)
        return importance_samples.reshape(actions.shape[0], 1)

    def _act(self, state):
        with torch.no_grad():
            values = self._networks.predict_values(
                torch.tensor(state, dtype=self._dtype, device=device).unsqueeze(0))[0]
        option = self._policy_train.act(values)
        return option, values.numpy()

    def prefill(self, env):
        state = env.reset()
        for step in tqdm.tqdm(range(self._prefill_steps)):
            action = np.random.randint(0, len(self._sub_policies), size=(1, ))
            next_state, rewards, done = self._sub_policies[action[0]].step(env, state, False)
            next_state = next_state if not done else None
            transition = [state, action, rewards.sum(), next_state]
            self._replay_buffer.remember(transition)
            state = next_state if not done else env.reset()

    def compute_values(self, states):
        with torch.no_grad():
            # Take actions w.r.t. the learning network
            actions = self._networks.predict_values(states).argmax(dim=1).unsqueeze(1)
            # Predict values of those actions with target network
            q_values = self._networks.predict_targets(states).gather(dim=1, index=actions)
        return q_values.numpy()


if __name__ == '__main__':
    N_STEPS_SUBPOLICY = 2
    LR_SUBPOLICY = 0.5e-3
    LR_MANAGER = 0.1e-3
    BATCH_SIZE_SUBPOLICY = 32
    BATCH_SIZE_MANAGER = 64
    BUFFER_LEN_SUBPOLICY = 10000
    PREFILL_SUBPOLICY_STEPS = 0
    BUFFER_LEN_MANAGER = 2000
    PREFILL_MANAGER_STEPS = 0
    TRAIN_STEPS = 100000
    TRAIN_INTERVAL = 100
    SHARED_REPLAY_BUFFER = False

    _env = gym.make('Pendulum-v0')
    _obs_len = _env.observation_space.shape[0]
    _act_len = _env.action_space.shape[0]
    _max_action = _env.action_space.high[0]

    # Create networks
    _actor = networks.LinearNetwork(
        inputs=_obs_len, outputs=_act_len, n_hidden_layers=1, n_hidden_units=128,
        activation_last_layer=torch.tanh, output_weight=_max_action)
    _critic = networks.LinearNetwork(
        inputs=_obs_len + _act_len, outputs=1, n_hidden_layers=1, n_hidden_units=128)
    _qnet = networks.LinearNetwork(
        inputs=_obs_len, outputs=2, n_hidden_layers=1, n_hidden_units=128)

    # Create hardcoded and RL subpolicies
    _replay_buffer = replay_buffers.FIFOReplayBuffer(maxlen=BUFFER_LEN_SUBPOLICY) if \
        SHARED_REPLAY_BUFFER else None
    hardcoded_sub_policy = HardcodedSubPolicy(
        policy=hardcoded_policies.pendulum, replay_buffer=_replay_buffer)
    rl_subpolicy = TD3SubPolicy(
        actor=_actor, critic=_critic, max_action=_max_action,
        min_action=-_max_action, buffer_len=BUFFER_LEN_SUBPOLICY, lr=LR_SUBPOLICY,
        prefill_steps=PREFILL_SUBPOLICY_STEPS, batch_size=BATCH_SIZE_SUBPOLICY,
        replay_buffer=_replay_buffer
    )

    hdq = HierarchicalDeepQ(
        q_net=_qnet, sub_policies=[hardcoded_sub_policy, rl_subpolicy],
        training_steps=TRAIN_STEPS, buffer_len=BUFFER_LEN_MANAGER, lr=LR_MANAGER,
        prefill_steps=PREFILL_MANAGER_STEPS, epsilon_start=0.2, epsilon_end=0.01,
        decay_schedule='lin', batch_size=BATCH_SIZE_MANAGER, train_interval=TRAIN_INTERVAL,
        alternate_training=True,
    )
    res = hdq.train(_env)

    from matplotlib import pyplot as plt
    plt.title('Rewards')
    plt.plot(res['end_steps'], res['rewards'], 'g', label='Rewards')
    plt.legend()
    plt.show()

    plt.title('Sub-policy use')
    plt.plot(res['end_steps'], res['subpolicy_use'][0], 'r', label='Hardcoded')
    plt.plot(res['end_steps'], res['subpolicy_use'][1], 'b', label='RL')
    plt.legend()
    plt.show()
