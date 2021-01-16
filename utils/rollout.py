import numpy as np
import torch
from torch.autograd import Variable


class RolloutWorker:
    def __init__(self, env, controller, config):
        self.env = env
        self.controller = controller
        self.n_agents = len(controller.n_agents)
        self.obs_shape = controller.obs_shape
        self.n_actions = controller.n_actions
        self.episode_limit = config.episode_limit
        self.config = config

        assert config.n_rollout_threads == 1, "This code only require one parallel environment!"

    def generate_episode(self, episode_num):
        o, u, r, u_onehot, terminated = [], [], [], [], []
        obs = self.env.reset()
        terminate = False
        step = 0
        episode_reward = 0
        self.controller.rin_net.init_hidden(1)

        while not terminate and step < self.episode_limit:
            # 这里需要考虑到粒子环境的多线程环境问题处理
            obs = obs.squeeze(0)  # 约束本环境仅能有1个并行环境，这里可以做降维
            actions = self.controller.select_actions(obs)
            actions_onehot = []
            for action in actions:
                action_onehot = np.zeros(self.n_actions)
                action_onehot[action] = 1
                actions_onehot.append(action_onehot)
            actions = actions.unsqueeze(0)
            obs_next, rewards, terminates, infos = self.env.step(actions)
            # 每个智能体收到的reward和terminate信息是一样的，按照buffer的设计，对reward和terminate降维处理
            rewards = rewards.reshape([-1, self.n_agents])
            reward = np.mean(rewards, axis=-1)[0]
            terminates = terminates.reshape([-1, self.n_agents])
            terminate = np.mean(terminates, axis=-1)[0]

            o.append(obs)
            u.append(np.reshape(actions, [self.n_agents, 1]))   # 降维
            r.append(reward)
            u_onehot.append(actions_onehot)
            terminated.append(terminate)

            episode_reward += reward
            step += 1
            obs = obs_next

        # Append the last infos
        obs = obs.reshape([self.n_agents, -1])
        o.append(obs)
        o_next = o[1:]
        o = o[:-1]

        # 封装成和buffer一样的字典类型
        episode = dict(o=o.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       o_next=o_next.copy(),
                       u_onehot=u_onehot.copy(),
                       terminated=terminated.copy()
                       )
        for key in episode.keys():
            episode[key] = np.array([episode[key]])

        return episode, episode_reward
