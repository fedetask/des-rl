import time

import gym
import numpy as np
import torch


def cartpole_naive(obs):
    x, v, theta, theta_dot = obs
    if theta < 0:
        return 0
    else:
        return 1


def cartpole_medium(obs):
    x, v, theta, theta_dot = obs
    if theta <= 0:
        if theta_dot > -0.05:
            return 1
        else:
            return 0
    else:
        if theta_dot < 0.05:
            return 0
        else:
            return 1


def cartpole_perfect(obs):
    x, v, theta, theta_dot = obs
    if theta <= 0:
        if theta_dot > 1:
            return 1
        else:
            return 0
    else:
        if theta_dot < -1:
            return 0
        else:
            return 1


def mountain_car_deterministic(obs):
    pos, vel = obs
    if vel < 0:
        return np.array([-1])
    else:
        return np.array([+1])


def mountain_car_explore(obs):
    pos, vel = obs
    if vel < 0:
        return np.array([-1])
    else:
        return None


def mountain_car_normal(obs):
    sigma = 1.0
    pos, vel = obs
    if vel < 0:
        return (np.random.randn(1) - 0.1) * sigma
    else:
        return (np.random.randn(1) + 0.1) * sigma


def pendulum(obs):
    cos, sin, vel = obs
    left, right = np.array([-1]), np.array([1])
    if vel >= 0:
        if cos <= 0:
            action = right
        else:
            action = left
    else:
        if cos <= 0:
            action = left
        else:
            action = right
    #print('State: ' + str(obs) + ' action: ' + str(action))
    return action


def eval_policy(policy, env, test_episodes=100, render=False, wait_key=False):
    rewards = []
    for e in range(test_episodes):
        state = env.reset()
        done = False
        tot_reward = 0
        while not done:
            if policy == 'random':
                action = env.action_space.sample()
            else:
                if isinstance(policy, torch.nn.Module):
                    with torch.no_grad():
                        action = policy(
                            torch.tensor(state).unsqueeze(0).float()
                        )[0].detach().numpy()
                else:
                    action = policy(state)
            next_state, rew, done, info = env.step(action)
            if render:
                env.render()
            if wait_key:
                input()
            tot_reward += rew
            state = next_state
        rewards.append(tot_reward)
    return np.array(rewards)


if __name__ == '__main__':
    pass
