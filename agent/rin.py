import torch
import torch.nn as nn

from utils.base_net import RNN, MixNet

TORCH_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class IntrinsicReward:
    def __init__(self, n_agents, n_actions, obs_shape, config):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.config = config
        # 采用循环神经网络，输入一个过去动作作为隐藏信息，输入一个n_agents的one-hot向量做多智能体的网络复用
        input_shape = self.obs_shape + self.n_actions + self.n_agents

        # Neural Network
        self.eval_rnn = RNN(input_shape, n_actions, config).to(TORCH_DEVICE)
        self.target_rnn = RNN(input_shape, n_actions, config).to(TORCH_DEVICE)
        self.eval_mix = MixNet().to(TORCH_DEVICE)
        self.target_mix = MixNet().to(TORCH_DEVICE)

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mix.load_state_dict(self.eval_mix.state_dict())
        self.eval_parameters = list(self.eval_mix.parameters()) + list(self.eval_rnn.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=config.lr)
        self.loss_func = nn.MSELoss()

        self.eval_hidden = None
        self.target_hidden = None

    def update(self, batch, episode_len, train_step):
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, terminated = batch['u'].to(TORCH_DEVICE), batch['r'].to(TORCH_DEVICE), batch['terminated'].to(TORCH_DEVICE)

        q_evals, q_targets = self.get_q_value(batch, episode_len)
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_targets = q_targets.max(dim=3)[0]

        q_total_eval = self.eval_mix(q_evals)
        q_total_target = self.target_mix(q_targets)

        targets = r + self.config.gamma * q_total_target * (1 - terminated)

        loss = self.loss_func(targets.detach(), q_total_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _get_inputs(self, batch, transition_idx):   # batch: Tensor
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        # 给obs添加上一个动作onehot、agent编号 (RNN)
        if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
            inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
        else:
            inputs.append(u_onehot[:, transition_idx - 1])
        inputs_next.append(u_onehot[:, transition_idx])

        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.n_agents个agent的数据拼成(episode_num * n_agents)条数据
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def get_q_value(self, batch, episode_len):  # batch: Tensor
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)   # 给obs加上last_action、agent_id
            # Transfer to CUDA if you have a GPU
            inputs = inputs.to(TORCH_DEVICE)
            inputs_next = inputs_next.to(TORCH_DEVICE)
            self.eval_hidden = self.eval_hidden.to(TORCH_DEVICE)
            self.target_hidden = self.target_hidden.to(TORCH_DEVICE)

            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # 把q_eval维度重新变回(episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

        # 得的q_eval和q_target是一个列表，列表里装着episode_len个数组，数组的的维度是(episode个数, n_agents, n_actions)
        # 把该列表转化成(episode_num, episode_len, n_agents, n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.config.rnn_hidden_dim)).to(TORCH_DEVICE)
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.config.rnn_hidden_dim)).to(TORCH_DEVICE)
