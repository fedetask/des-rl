import torch
from torch import nn
import gym
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import json

from deepq import computations, deepqnetworks, policies, replay_buffers, utils
import experiment_utils
import hardcoded_policies


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
        x = self.activation(self.in_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.out_layer(x)
        return x


def plot_episode_results(episode_lengths, losses, episode_end_steps, argmax_q_values, epsilons,
                         avg_targets):
    print('Training completed in ' + str(len(episode_lengths)) + ' episodes')
    plt.plot(range(len(losses)), losses, label='Loss')
    plt.plot(episode_end_steps, episode_lengths, label='Episode lengths')
    plt.plot(range(len(argmax_q_values)), argmax_q_values,
             label='Q value for argmax action')
    plt.plot(range(len(epsilons)), epsilons, label='Epsilon')
    plt.plot(range(len(avg_targets)), avg_targets, label='Average target value')
    plt.legend()
    plt.show()


def test_agent(q_network, env, num_episodes):
    q_network.eval()
    policy_test = policies.GreedyPolicy()
    episode_rewards = [0]
    state = env.reset()
    for i in range(num_episodes):
        while True:
            with torch.no_grad():
                q_values = q_network.predict_values(state)
                action = policy_test.act(q_values)[0]
            next_state, reward, done, _ = env.step(action)
            episode_rewards[-1] += reward
            if done:
                state = env.reset()
                episode_rewards.append(0)
            else:
                state = next_state
    print('Average total reward over ' + str(num_episodes) + ' episodes: ' +
          str(np.mean(episode_rewards)))


def run_experiment(p_hardcoded_start, p_hardcoded_schedule,
                   p_hardcoded_end, hardcoded_policy=None, training_steps=4000, pretrain_steps=1000,
                   update_net_steps=10, batch_size=128, df=0.99, lr=0.5e-3, buffer_len=10000,
                   epsilon_start=1.0, epsilon_end=0.1, epsilon_schedule='exp', decay_steps=None):
    if decay_steps is None:
        decay_steps = training_steps
    if p_hardcoded_schedule == 'lin_decay':
        assert p_hardcoded_start >= p_hardcoded_end
    if p_hardcoded_schedule == 'lin_upcay':
        assert p_hardcoded_end >= p_hardcoded_start
    p_hardcoded = p_hardcoded_start

    env = gym.make('CartPole-v0')

    # Replay buffer and prefiller
    replay_buffer = replay_buffers.FIFOReplayBuffer(maxlen=buffer_len)
    prefiller = replay_buffers.UniformGymPrefiller()

    # Q Networks
    q_net = QNet(state_shape=4, act_shape=2, n_layers=3, n_units=16)
    dq_nets = deepqnetworks.TargetDQNetworks(q_network=q_net)

    # Target computer and trainer
    target_computer = computations.DoubleQTargetComputer(dq_networks=dq_nets, df=df)
    trainer = computations.DQNTrainer(dq_networks=dq_nets, lr=lr)
    if epsilon_schedule == 'exp_decay':
        policy_train = policies.ExponentialDecayEpsilonGreedy(decay_steps=decay_steps,
                                                              start_epsilon=epsilon_start,
                                                              end_epsilon=epsilon_end)
    elif epsilon_schedule == 'lin_decay':
        policy_train = policies.LinearDecayEpsilonGreedyPolicy(decay_steps=decay_steps,
                                                               start_epsilon=epsilon_start,
                                                               end_epsilon=epsilon_end)
    elif epsilon_schedule == 'const':
        policy_train = policies.FixedEpsilonGreedyPolicy(epsilon=epsilon_start)

    prefiller.fill(replay_buffer, env, pretrain_steps, shuffle=False)

    # Training
    losses = []
    rewards = [0]
    episode_end_steps = []
    argmax_q_values = []
    epsilons = []
    avg_targets = []

    episode_steps = 0
    state = env.reset()
    for step in tqdm(range(training_steps)):
        with torch.no_grad():
            q_values = dq_nets.predict_values(state)
            if np.random.binomial(1, p=p_hardcoded):
                action = hardcoded_policy(state)
            else:
                action = policy_train.act(q_values)[0]
            if p_hardcoded_schedule == 'lin_decay':
                p_hardcoded = p_hardcoded_start - step * \
                              (p_hardcoded_start - p_hardcoded_end) / training_steps
            elif p_hardcoded_schedule == 'lin_upcay':
                p_hardcoded = p_hardcoded_start + \
                              step * (p_hardcoded_end - p_hardcoded_start) / training_steps
        argmax_q_values.append(float(q_values.numpy().max()))
        epsilons.append(policy_train.cur_epsilon)
        next_state, reward, done, _ = env.step(action)
        if done:
            next_state = None
        replay_buffer.remember((state, action, reward, next_state))
        rewards[-1] += reward

        if len(replay_buffer.buffer) >= batch_size:
            batch = replay_buffer.sample(size=batch_size)
            batch = utils.split_replay_batch(batch)
            targets = target_computer.compute_targets(batch)
            avg_targets.append(float(targets.mean()))
            loss, grads = trainer.train(batch, targets)
            losses.append(float(loss))
            if step % update_net_steps == 0:
                dq_nets.update()
        else:
            avg_targets.append(0)
            losses.append(0)

        episode_steps += 1
        if done:
            state = env.reset()
            rewards.append(0)
            episode_end_steps.append(step)
            episode_steps = 0
        else:
            state = next_state
    del rewards[-1]
    return {
        'rewards': rewards,
        'end_steps': episode_end_steps,
        'losses': losses,
        'argmax_q': argmax_q_values,
        'epsilons': epsilons,
        'avg_targets': avg_targets,
        'parameters': {
            'training_steps': training_steps,
            'pretrain_steps': pretrain_steps,
            'buffer_len': buffer_len,
            'update_net_steps': update_net_steps,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay_steps': decay_steps,
            'lr': lr,
            'batch_size': batch_size,
            'df': df,
            'p_hardcoded_start': p_hardcoded_start,
            'p_hardcoded_end': p_hardcoded_end,
            'p_hardcoded_schedule': p_hardcoded_schedule,
        }
    }


if __name__ == '__main__':
    # Only these values should be modified across experiments
    p_hardcoded_starts = [0.1, 0.2, 0.3, 0.4, 0.5]
    p_hardcoded_ends = np.zeros_like(p_hardcoded_starts)

    p_hardcoded_schedule = 'lin_decay'
    NUM_EXPERIMENTS = 20

    results_folder = 'experiment_results/policy_cartpole_medium/'
    file_prefix = 'const_epsilon'
    hardcoded_policy = hardcoded_policies.cartpole_medium
    epsilon_start = 0.2
    epsilon_end = 0.0
    epsilon_schedule = 'const'

    for p_hardcoded_start, p_hardcoded_end in zip(p_hardcoded_starts, p_hardcoded_ends):
        p_hardcoded_start = round(p_hardcoded_start, 1)
        p_hardcoded_end = round(p_hardcoded_end, 1)
        exp_name = file_prefix + p_hardcoded_schedule
        if p_hardcoded_start != p_hardcoded_end:
            exp_name += '_' + str(p_hardcoded_start) + '_' + str(p_hardcoded_end)
        else:
            exp_name += '_' + str(p_hardcoded_start)

        exp_losses = []
        exp_rewards = []
        exp_episode_end_steps = []
        exp_argmax_q_values = []
        exp_epsilons = []
        exp_avg_targets = []

        for e in range(NUM_EXPERIMENTS):
            res = run_experiment(p_hardcoded_start=p_hardcoded_start,
                                 p_hardcoded_end=p_hardcoded_end,
                                 p_hardcoded_schedule=p_hardcoded_schedule,
                                 hardcoded_policy=hardcoded_policy,
                                 epsilon_start=epsilon_start,
                                 epsilon_end=epsilon_end,
                                 epsilon_schedule=epsilon_schedule)
            exp_rewards.append(res['rewards'])
            exp_episode_end_steps.append(res['end_steps'])
            exp_losses.append(res['losses'])
            exp_avg_targets.append(res['avg_targets'])
            exp_argmax_q_values.append(res['argmax_q'])
            exp_epsilons.append(res['epsilons'])

        experiment_data = {
            'n_runs': NUM_EXPERIMENTS,
            'p_hardcoded_schedule': p_hardcoded_schedule,
            'exp_rewards': exp_rewards,
            'exp_losses': exp_losses,
            'exp_end_steps': exp_episode_end_steps,
            'exp_avg_targets': exp_avg_targets,
            'exp_q_values': exp_argmax_q_values,
            'exp_epsilons': exp_epsilons,
            'parameters': res['parameters'],
        }

        reward_plots = [
            np.array([end_steps, lengths])
            for end_steps, lengths in zip(exp_episode_end_steps, exp_rewards)
        ]

        x_avg, y_avg, var, y_arrays = experiment_utils.merge_plots(reward_plots)

        json_content = json.dumps(experiment_data)
        f = open(results_folder + exp_name, 'w')
        f.write(json_content)
        f.close()
        print('Written ' + results_folder + exp_name)
