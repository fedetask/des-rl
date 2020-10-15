import gym
import numpy as np
import time


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
    left, right = np.array([-2]), np.array([2])
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


if __name__ == '__main__':
    TEST_EPISODES = 100

    env = gym.make('Pendulum-v0')
    policy = pendulum

    rewards = [0]
    target_reached = 0
    for e in range(TEST_EPISODES):
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, rew, done, info = env.step(action)
            #env.render()
            #input()
            rewards[-1] += rew
            target_reached += rew > 0
            state = next_state
            if done:
                rewards.append(0)
    del rewards[-1]
    env.close()
    print('Average reward over ' + str(TEST_EPISODES) + ': ' + str(np.mean(rewards)))
    print('Goal reached in ' + str(target_reached) + ' episodes over ' + str(TEST_EPISODES))