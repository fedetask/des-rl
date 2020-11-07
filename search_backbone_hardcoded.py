import gym
import torch

from experiment_residual import backbone_training
import hardcoded_policies


if __name__ == '__main__':
    NUM_RUNS = 10
    TRAINING_STEPS = 15000
    BUFFER_LEN = 100000
    BUFFER_PREFILL = 2000
    CRITIC_LR = 1e-3
    ACTOR_LR = 1e-3
    UPDATE_NET_EVERY = 2
    DISCOUNT_FACTOR = 0.99
    BATCH_SIZE = 100
    EPSILON_START = 0.2
    EPSILON_END = 0.0
    EPSILON_DECAY_SCHEDULE = 'lin'
    CHECKPOINT_EVERY = 1000

    _env = gym.make('Pendulum-v0')

    for prefill_nosie in [0.05, 0.2, 0.5, 1.0, 2.0]:
        for eps_start in [0.2, 0.1, 0.05]:
            backbone_training(
                env=_env, train_steps=TRAINING_STEPS, num_runs=NUM_RUNS,
                backbone_policy=hardcoded_policies.pendulum, buffer_len=BUFFER_LEN,
                buffer_prefill=BUFFER_PREFILL, df=DISCOUNT_FACTOR, actor_lr=ACTOR_LR,
                critic_lr=CRITIC_LR, batch_size=BATCH_SIZE, eps_start=eps_start,
                eps_end=EPSILON_END, eps_decay=EPSILON_DECAY_SCHEDULE,
                collection_policy_noise=prefill_nosie, checkpoint_every=CHECKPOINT_EVERY,
                results_dir=f'experiment_results/td3/backbone/{_env.unwrapped.spec.id}/',
                exp_name_suffix=f'_prefill_noise_{prefill_nosie}_eps_start_{eps_start}_hardcoded'
            )
