import gym
import torch

from experiment_residual import continue_training

if __name__ == '__main__':
    NUM_RUNS = 10
    TRAINING_STEPS = 15000
    BUFFER_PREFILL = 1000
    BUFFER_LEN = 100000
    CRITIC_LR = 1e-3
    ACTOR_LR = 1e-3
    UPDATE_NET_EVERY = 2
    DF = 0.99
    BATCH_SIZE = 100
    EPSILON_START = 0.2
    EPSILON_END = 0.0
    EPSILON_DECAY_SCHEDULE = 'lin'
    COLLECTION_POLICY_NOISE = EPSILON_START
    CHECKPOINT_EVERY = 1000

    _env = gym.make('Pendulum-v0')

    actor_net = torch.load('models/standard/Pendulum-v0/actor_9000')
    critic_net = torch.load('models/standard/Pendulum-v0/critic_9000')[0]

    def model_policy(state):
        with torch.no_grad():
            action = actor_net(
                torch.tensor(state).unsqueeze(0).float()
            )[0].detach().numpy()
        return action

    prefill_values = [0, 2000]
    for prefill in prefill_values:
        continue_training(
            env=_env, train_steps=TRAINING_STEPS, num_runs=NUM_RUNS, actor_net=actor_net,
            critic_net=critic_net, buffer_len=BUFFER_LEN, buffer_prefill=prefill,
            actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, df=DF, batch_size=BATCH_SIZE,
            eps_start=EPSILON_START, eps_end=EPSILON_END, eps_decay=EPSILON_DECAY_SCHEDULE,
            collection_policy=model_policy, collection_policy_noise=COLLECTION_POLICY_NOISE,
            checkpoint_every=CHECKPOINT_EVERY,
            results_dir=f'experiment_results/td3/continue/{_env.unwrapped.spec.id}/',
            exp_name_suffix=f'_prefill_{prefill}_collection_noise_{COLLECTION_POLICY_NOISE}'
                            f'_eps_{EPSILON_START}_to_{EPSILON_END}_{EPSILON_DECAY_SCHEDULE}'
        )