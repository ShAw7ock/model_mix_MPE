import torch
import numpy as np
from gym.spaces import Tuple, Box, Discrete

from agent.mpc import MpcAgent
from agent.rin import IntrinsicReward

TORCH_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Controller:
    def __init__(self, agent_init_params, algo_types, config):
        # 使用MPE里三个智能体全部同构的条件
        self.n_agents = len(algo_types)
        self.n_actions = agent_init_params[0]['action_space']
        self.obs_shape = agent_init_params[0]['observation_space']
        self.algo_types = algo_types
        self.rin_net = IntrinsicReward(self.n_agents, self.n_actions, self.obs_shape, config)
        self.agents = [MpcAgent(self.n_agents, agent_id, params['observation_space'], params['action_space'],
                                params['observation_dim'], params['action_dim'],
                                config)
                       for agent_id, params in enumerate(agent_init_params)]
        self.agent_init_params = agent_init_params
        self.config = config

    def select_actions(self, observations):
        actions = []
        for agent, obs in zip(self.agents, observations):
            if not agent.has_been_trained:
                action = np.random.choice(range(self.n_actions))
                actions.append(action)
                continue
            if agent.act_buffer.shape[0] > 0:
                # 这一步会选出最优的动作选出，并把self.act_buffer清空
                action, agent.act_buffer = agent.act_buffer[0], agent.act_buffer[1:]
            else:
                agent.sy_cur_obs = obs
                soln = agent.optimizer.obtain_solution(agent.prev_sol, agent.init_var, self._compile_acc_rin)
                # agent.per 决定action sequence多久更新一个，故动作序列中的动作个数为 agent.per * agent.dU (agent.per = 1)
                agent.prev_sol = np.concatenate([np.copy(soln)[agent.per * agent.act_dim:],
                                                 np.zeros(agent.per * agent.act_dim)])
                agent.act_buffer = soln[:agent.per * agent.act_dim].reshape(-1, agent.act_dim)
                action, agent.act_buffer = agent.act_buffer[0], agent.act_buffer[1:]
            actions.append(action)

        return actions

    @torch.no_grad()
    def _compile_acc_rin(self, act_seqs, agent_num):
        cur_agent = self.agents[agent_num]
        num_seq = act_seqs.shape[0]
        # 处理action sequence的类型和维度
        act_seqs = torch.from_numpy(act_seqs).float().to(TORCH_DEVICE)
        act_seqs = act_seqs.view(-1, cur_agent.task_horizon, cur_agent.act_dim)
        transposed = act_seqs.transpose(0, 1)
        expanded = transposed[:, :, None]
        tiled = expanded.expand(-1, -1, cur_agent.n_particles, -1)
        act_seqs = tiled.contiguous().view(cur_agent.task_horizon, -1, cur_agent.act_dim)

        # 处理current observation的维度
        cur_obs = torch.from_numpy(cur_agent.sy_cur_obs).float().to(TORCH_DEVICE)
        cur_obs = cur_obs[None]     # 增加一维
        cur_obs = cur_obs.expand(num_seq * cur_agent.n_particles, -1)   # 将current observation扩展到每一条轨迹

        # 增一个表示agent序号的onehot向量，用以RNN的输入inputs
        agent_id = torch.zeros(self.n_agents).to(TORCH_DEVICE)
        agent_id[agent_num] = 1
        agent_id = agent_id[None]
        agent_id = agent_id.expand(num_seq * cur_agent.n_particles, -1)

        # 复制一份该agent的eval_hidden，用于model想象的hidden_state不应该影响到真实环境train种的hidden_state更新
        hidden_state = self.rin_net.eval_hidden[:, agent_num, :]

        returns = torch.zeros(num_seq, cur_agent.n_particles, device=TORCH_DEVICE)

        # t=0 ~ t=task_horizon属于model预测的想象部分
        for t in cur_agent.task_horizon:
            cur_act = act_seqs[t]

            # 处理last_action用作RNN的输入inputs一部分
            last_act_onehot = torch.zeros(cur_agent.act_space).to(TORCH_DEVICE)
            if t > 0:
                last_act_onehot[act_seqs[t - 1]] = 1
            last_act_onehot = last_act_onehot[None]
            last_act_onehot = last_act_onehot.expand(num_seq * cur_agent.n_particles, -1)

            inputs = torch.cat((cur_obs, last_act_onehot, agent_id), dim=-1)

            rin, hidden_state = self.rin_net.eval_rnn(inputs, hidden_state)
            rin = torch.gather(rin, dim=1, index=cur_act)
            rin = rin.view(-1, cur_agent.n_particles)

            returns += rin

            cur_obs = self._predict_next_obs(cur_obs, cur_act, agent_num)

        return returns.mean(dim=1).detach().cpu().numpy()

    def _predict_next_obs(self, obs, act, agent_num):
        cur_agent = self.agents[agent_num]

        obs = cur_agent.expand_to_model_format(obs)
        act = cur_agent.expand_to_model_format(act)
        inputs = torch.cat((obs, act), dim=-1)

        mean, var = cur_agent.models(inputs)

        predictions = mean + torch.randn_like(mean, device=TORCH_DEVICE) * var.sqrt()

        predictions = cur_agent.flatten_to_matrix(predictions)

        return predictions

    @classmethod
    def init_from_env(cls, env, config, agent_algo="MADDPG", adversary_algo="MADDPG"):
        agent_init_params = []

        algo_types = [adversary_algo if agent_type == 'adversary' else agent_algo
                      for agent_type in env.agent_types]

        def get_shape(sp):  # Get the shape of action spaces and action dim
            shape, dim = 0, 0
            if isinstance(sp, Box):
                shape = sp.shape[0]
                dim = sp.shape[0]
            elif isinstance(sp, Tuple):
                for p in sp.spaces:
                    if isinstance(p, Box):
                        shape += p.shape[0]
                        dim += p.shape[0]
                    else:
                        shape += p.n
                        dim += 1
            else:  # if the instance is 'Discrete', the action dim is 1
                shape = sp.n
                dim = 1
            return shape, dim

        for acsp, obsp in zip(env.action_space, env.observation_space):
            observation_space, observation_dim = get_shape(obsp)
            action_space, action_dim = get_shape(acsp)
            agent_init_params.append({'observation_space': observation_space,
                                      'observation_dim': observation_dim,
                                      'action_space': action_space,
                                      'action_dim': action_dim})

        init_dict = {'config': config,
                     'algo_types': algo_types,
                     'agent_init_params': agent_init_params}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
