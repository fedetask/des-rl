import torch

from deepq import replay_buffers
from deepq import computations
from deepq import policies
from deepq import deepqnetworks


class DQN:

    def __init__(self, q_net, training_steps, buffer_len=100000, df=0.99,
                 batch_size=128, lr=0.5e-3, update_targets_every=2, update_target_mode='hard',
                 tau=0.005, epsilon_start=0.15, epsilon_end=0.01,
                 epsilon_decay_schedule='const', dtype=torch.float, evaluate_every=-1,
                 evaluation_episodes=5, checkpoint_every=-1, checkpoint_dir=None):
        self._training_steps = training_steps
        self._df = df
        self._batch_size = batch_size
        self._lr = lr
        self._update_targets_every = update_targets_every
        self._update_target_mode = update_target_mode
        self._tau = tau
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_schedule = epsilon_decay_schedule
        self._dtype = dtype
        self._evaluate_every = evaluate_every
        self._evaluation_episodes = evaluation_episodes
        self._checkpoint_every = checkpoint_every
        self._checkpoint_dir = checkpoint_dir

        self.q_networks = deepqnetworks.TargetDQNetworks(q_network=q_net)
        self.replay_buffer = replay_buffers.FIFOReplayBuffer(maxlen=buffer_len)
        self.target_computer = computations.DoubleQTargetComputer(
            dq_networks=self.q_networks, df=df, dtype=dtype)
        self.trainer = computations.DQNTrainer(dq_networks=self.q_networks, lr=lr, dtype=dtype)
        self.policy_train = policies.EpsilonGreedyPolicy(
            start_epsilon=epsilon_start, end_epsilon=epsilon_end, decay_steps=training_steps,
            decay_schedule=epsilon_decay_schedule)


