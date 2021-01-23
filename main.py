import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Tuple, Discrete
from pathlib import Path
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.rollout import RolloutWorker
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algos.controller import Controller
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


def train_your_mode():
    run(config)


def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action=True):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def run(config):
    model_dir = Path('./results') / config.env_id
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir()
                         if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)

    run_dir = model_dir / curr_run
    fig_dir = run_dir / 'figures'
    os.makedirs(str(fig_dir))
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    assert config.n_rollout_threads == 1, "For simple test, we assume the number of the environment is 1"
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed)

    controller = Controller.init_from_env(env=env, config=config)
    obs_shape, n_actions = controller.obs_shape, controller.n_actions
    buffer = ReplayBuffer(controller.n_agents, obs_shape, n_actions, config.episode_limit, config.buffer_size)
    rolloutworker = RolloutWorker(env, controller, config)

    train_step = 0
    mean_episode_rewards = []
    for ep_i in range(config.n_episodes):
        episode, ep_rew, mean_ep_rew = rolloutworker.generate_episode()
        buffer.push(episode)
        for step in range(config.n_train_steps):
            mini_batch = buffer.sample(min(len(buffer), config.batch_size))
            controller.update(mini_batch, train_step)
            train_step += 1
        # ep_rew = buffer.get_average_rewards(config.episode_limit * config.n_rollout_threads)
        mean_episode_rewards.append(mean_ep_rew)
        print("Episode {} : Total reward {} , Mean reward {}" .format(ep_i + 1, ep_rew, mean_ep_rew))

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(str(run_dir / 'incremental'), exist_ok=True)
            controller.save(str(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1))))
            controller.save(str(run_dir / 'model.pt'))

    controller.save(str(run_dir / 'model.pt'))
    env.close()

    index = list(range(1, len(mean_episode_rewards) + 1))
    plt.plot(index, mean_episode_rewards)
    plt.ylabel("Mean Episode Reward")
    plt.savefig(str(fig_dir) + '/mean_episode_reward.jpg')
    # plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Normal Setting
    parser.add_argument("--env_id", default='simple_spread', type=str,
                        help="Name of environment")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int, help="For simple test, we assume here to be 1")
    parser.add_argument("--n_episodes", default=1000, type=int, help="Total episodes to train")
    parser.add_argument("--n_train_steps", default=1, type=int, help="Training steps with buffer storage")
    parser.add_argument("--save_interval", default=100, type=int, help="The step interval between the saving model")
    parser.add_argument("--display", default=False, type=bool, help="Choose to render while training")

    # Arguments for Intrinsic Reward Network
    parser.add_argument("--episode_limit", default=25, type=int, help="The length of each episode sampling length")
    parser.add_argument("--buffer_size", default=int(200), type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--rnn_hidden_dim", default=64, type=int)

    # Arguments for MPC Model initial config
    parser.add_argument("--per", default=1, type=int,
                        help="Determines how often the action sequence will be optimized.")
    parser.add_argument("--num_models", default=5, type=int, help="size of model ensemble")
    parser.add_argument("--train_epochs", default=5, type=int, help="numbers of updating the model")
    parser.add_argument("--mode", default='TSinf', type=str)
    parser.add_argument("--n_particles", default=20, type=int, help="numbers of particles while model sampling")
    parser.add_argument("--ign_var", default=False, type=bool)
    parser.add_argument("--load_models", default=None, type=str, help="Load the saved models have been trained")
    parser.add_argument("--model_pretrained", default=False, type=bool)

    # Arguments for MPC Model optimizer config
    parser.add_argument("--task_horizon", default=10, type=int, help="The forward steps using the internal models")
    parser.add_argument("--opt_alpha", default=0.1, type=float, help="The weight while updating the optimizer")
    parser.add_argument("--opt_max_iters", default=5, type=int)
    parser.add_argument("--opt_num_elites", default=40, type=int, help="The number to choose to update the optimizer")
    parser.add_argument("--opt_popsize", default=100, type=int, help="The total number of optimizer sampling")

    config = parser.parse_args()

    train_your_mode()
