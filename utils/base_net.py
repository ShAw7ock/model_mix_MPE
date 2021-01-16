import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, config):
        super(RNN, self).__init__()
        self.args = config

        self.fc1 = nn.Linear(input_shape, config.rnn_hidden_dim)
        self.rnn = nn.GRUCell(config.rnn_hidden_dim, config.rnn_hidden_dim)
        self.fc2 = nn.Linear(config.rnn_hidden_dim, config.n_actions)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class MixNet(nn.Module):
    def __init__(self):
        super(MixNet, self).__init__()

    def forward(self, q_values):
        return torch.sum(q_values, dim=2, keepdim=True)
