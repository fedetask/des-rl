import torch
from torch import nn
import gym
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

from core.deepq import utils
from core.deepq import policies
from core.deepq import computations
from core.deepq import deepqnetworks
from core.deepq import replay_buffers


def handcoded_policy(obs):
    x, v, theta, theta_dot = obs
    if theta < 0:
        return 0
    else:
        return 1


class QNet(nn.Module):
    """Just a Q Network class to test the components
    """

    def __init__(self, state_shape, act_shape, n_layers, n_units, activation=torch.tanh):
        super().__init__()
        self.state_shape = state_shape
        self.act_shape = act_shape
        self.activation = activation

        self.in_layer = nn.Linear(in_features=state_shape, out_features=n_units)
        self.hidden_layers = [nn.Linear(n_units, n_units) for i in range(n_layers - 2)]
        self.out_layer = nn.Linear(in_features=n_units, out_features=act_shape)

    def forward(self, x):
        x = self.in_layer(x)
        for layer in self.hidden_layers:
            x = self.activation(x)
            x = layer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        return x


if __name__ == '__main__':
    TRAINING_STEPS = 5000
    PRETRAIN_STEPS = 1000
    UPDATE_NET_STEPS = 10
    BATCH_SIZE = 128
    GAMMA = 0.99
    LEARNING_RATE = 0.5e-3
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    DECAY_STEPS = TRAINING_STEPS
    TEST_EPISODES = 100

    env = gym.make('CartPole-v0')

    # Replay buffer and prefiller
    replay_buffer = replay_buffers.FIFOReplayBuffer(
        maxlen=10000, state_shape=(4, ), action_shape=(1, )
    )
    prefiller = replay_buffers.UniformGymPrefiller()

    # Q Networks
    q_net = QNet(state_shape=4, act_shape=2, n_layers=2, n_units=16)
    t_net = QNet(state_shape=4, act_shape=2, n_layers=2, n_units=16)
    dq_nets = deepqnetworks.TargetDQNetworks(q_network=q_net, target_network=t_net)

    # Target computer and trainer
    target_computer = computations.DoubleQTargetComputer(df=GAMMA)
    optimizer = torch.optim.RMSprop(dq_nets.get_trainable_params())
    trainer = computations.DQNTrainer(dq_networks=dq_nets, lr=LEARNING_RATE)
    policy_train = policies.ExponentialDecayEpsilonGreedy(decay_steps=DECAY_STEPS,
                                                          start_epsilon=EPSILON_START,
                                                          end_epsilon=EPSILON_END)

    prefiller.fill(replay_buffer, env, PRETRAIN_STEPS, shuffle=True)

    # Training
    losses = []
    episode_lengths = []
    episode_end_steps = []
    argmax_q_values = []
    epsilons = []
    avg_targets = []
    episode_steps = 0

    state = env.reset()
    for step in tqdm(range(TRAINING_STEPS)):
        with torch.no_grad():
            q_values = dq_nets.predict_values(torch.Tensor(state).float())
            action = policy_train.act(q_values)[0]
        argmax_q_values.append(q_values.numpy().max())
        epsilons.append(policy_train.cur_epsilon)
        next_state, reward, done, _ = env.step(action)
        if done:
            next_state = None
        replay_buffer.remember((state, action, reward, next_state))
        if len(replay_buffer.buffer) < BATCH_SIZE:
            continue
        batch = replay_buffer.sample(size=BATCH_SIZE)
        batch = utils.split_replay_batch(batch)
        targets = target_computer.compute_targets(dq_nets, batch)
        avg_targets.append(targets.mean())
        loss, grads = trainer.train(batch, targets)
        losses.append(loss)
        if step % UPDATE_NET_STEPS == 0:
            dq_nets.update_hard()

        episode_steps += 1

        if done:
            state = env.reset()
            episode_lengths.append(episode_steps)
            episode_end_steps.append(step)
            episode_steps = 0
        else:
            state = next_state

    print('Training completed in ' + str(len(episode_lengths)) + ' episodes')
    plt.plot(range(len(losses)), losses, label='Loss')
    plt.plot(episode_end_steps, episode_lengths, label='Episode lengths')
    plt.plot(range(len(argmax_q_values)), argmax_q_values, label='Q value for argmax action')
    plt.plot(range(len(epsilons)), epsilons, label='Epsilon')
    plt.plot(range(len(avg_targets)), avg_targets, label='Average target value')
    plt.legend()
    plt.show()

    dq_nets.q_network.eval()
    policy_test = policies.GreedyPolicy()
    episode_rewards = [0]
    state = env.reset()
    while True:
        with torch.no_grad():
            q_values = dq_nets.predict_values(state)
            action = policy_test.act(q_values)[0]
        next_state, reward, done, _ = env.step(action)
        episode_rewards[-1] += reward

        if done:
            state = env.reset()
            if len(episode_rewards) == TEST_EPISODES:
                break
            episode_rewards.append(0)
        else:
            state = next_state
    print('Average total reward over ' + str(TEST_EPISODES) + ' episodes: ' +
          str(np.mean(episode_rewards)))
