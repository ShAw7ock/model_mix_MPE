import os
import numpy as np
import torch
from tqdm import trange

from mpc_utils.optimizer import CEMOptimizer
from mpc_utils.dynamic_model import nn_constructor

TORCH_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def shuffle_rows(arr):
    # np.argsort(a, axis=-1, kind='quicksort', order=None) 按a中的元素按axis方向从小到大排序后，提取对应的索引index
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


# This class builds the agent for a single one.
class MpcAgent:
    def __init__(self, n_agents, agent_num, observation_space, action_space,
                 observation_dim, action_dim,
                 params):
        # config.pprint()
        self.n_agents = n_agents
        self.agent_num = agent_num
        self.obs_space, self.act_space = observation_space, action_space
        self.obs_dim, self.act_dim = observation_dim, action_dim
        self.params = params
        # self.act_ub, self.act_lb = params.env.action_space.high, params.env.action_space.low
        self.per = params.per

        # Model initial config
        self.model_input_dim = observation_dim + action_dim
        self.model_output_dim = observation_dim
        self.num_models = params.num_models
        self.model_train_epochs = params.train_epochs
        self.batch_size = params.batch_size
        self.prop_mode = params.mode
        self.n_particles = params.n_particles
        self.ign_var = params.ign_var or self.prop_mode == 'E'
        self.load_models = params.load_models

        # Lambda function
        self.obs_preproc = lambda obs: obs
        self.obs_postproc = lambda obs, model_out: model_out
        self.obs_postproc2 = lambda next_obs: next_obs
        self.targ_proc = lambda obs, next_obs: next_obs

        # Optimizer initial config
        self.task_horizon = params.task_horizon
        self.opt_alpha = params.opt_alpha
        self.opt_max_iters = params.opt_max_iters
        self.opt_num_elites = params.opt_num_elites
        self.opt_popsize = params.opt_popsize

        assert self.prop_mode == 'TSinf', 'Only TSinf propagation mode is supported'
        assert (self.n_particles % self.num_models == 0), "Number of particles must be a multiple of the ensemble size"

        # Create action sequence optimizer
        self.optimizer = CEMOptimizer(
            agent_num=agent_num, sol_dim=self.task_horizon * self.act_dim, max_iters=self.opt_max_iters,
            popsize=self.opt_popsize, num_elites=self.opt_num_elites, alpha=self.opt_alpha
        )

        # Agent Controller variables
        self.sy_cur_obs = None
        self.has_been_trained = params.model_pretrained
        self.act_buffer = np.array([]).reshape(0, self.act_dim)
        # TODO: 初始采样的mean和var在这里看一下还需要改，因为动作是Discrete表示的，没有 upper_bound 和 lower_bound
        self.prev_sol = np.tile(0, [self.task_horizon])
        self.init_var = np.tile(1, [self.task_horizon])
        # self.train_in plays the role of buffer to store (s,a)
        self.train_in = np.array([]).reshape(0, self.act_dim + self.obs_preproc(np.zeros([1, self.obs_dim])).shape[-1])
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.obs_dim]), np.zeros([1, self.obs_dim])).shape[-1]
        )

        # Set up the local model for each agent
        self.models = nn_constructor(num_models=self.num_models,
                                     input_dim=self.model_input_dim,
                                     output_dim=self.model_output_dim)

    def model_transition_train(self, batch):
        """
        Trains the internal model of transition function T(o'|o, a).
        Once trained, this agent switches from applying random actions to using MPC.

        Below Arguments:
            obs: array of observations. (n, obs_dim)
            obs_next: array of next observations. (n, obs_dim)
            actions: array of actions. (n, action_dim)

        Returns: None.
        """
        obs, obs_next, actions = batch['o'][:, :, self.agent_num, :], batch['o_next'][:, :, self.agent_num, :],\
                                 batch['u'][:, :, self.agent_num, :]
        obs = np.reshape(obs, [-1, self.obs_dim])
        obs_next = np.reshape(obs_next, [-1, self.obs_dim])
        actions = np.reshape(actions, [-1, self.act_dim])

        new_train_in = np.concatenate([obs, actions], axis=-1)
        new_train_targs = obs_next

        self.train_in = np.concatenate([self.train_in, new_train_in], axis=0)
        self.train_targs = np.concatenate([self.train_targs, new_train_targs], axis=0)

        # Change the signal of pretrained
        self.has_been_trained = True

        # Fit the input mean and variance
        self.models.fit_input_stats(self.train_in)

        idxs = np.random.randint(self.train_in.shape[0], size=[self.models.num_models, self.train_in.shape[0]])
        epoch_range = trange(self.model_train_epochs, unit="epoch(s)", desc="Network training")
        num_batch = int(np.ceil(idxs.shape[-1] / self.batch_size))

        for _ in epoch_range:
            for batch_num in range(num_batch):
                batch_idxs = idxs[:, batch_num * self.batch_size:(batch_num + 1) * self.batch_size]

                loss = 0.01 * (self.models.max_logvar.sum() - self.models.min_logvar.sum())
                loss += self.models.compute_decays()

                train_in = torch.from_numpy(self.train_in[batch_idxs]).to(TORCH_DEVICE).float()
                train_targ = torch.from_numpy(self.train_targs[batch_idxs]).to(TORCH_DEVICE).float()

                mean, logvar = self.models(train_in, ret_logvar=True)
                inv_var = torch.exp(-logvar)

                train_losses = ((mean - train_targ) ** 2) * inv_var + logvar
                train_losses = train_losses.mean(-1).mean(-1).sum()

                loss += train_losses

                self.models.optim.zero_grad()
                loss.backward()
                self.models.optim.step()

            # 打乱idxs的顺序，每一个epoch重新选取用作训练的idxs
            idxs = shuffle_rows(idxs)

            val_in = torch.from_numpy(self.train_in[idxs[:5000]]).to(TORCH_DEVICE).float()
            val_targ = torch.from_numpy(self.train_targs[idxs[:5000]]).to(TORCH_DEVICE).float()

            mean, _ = self.models(val_in)
            mse_losses = ((mean - val_targ) ** 2).mean(-1).mean(-1)

            epoch_range.set_postfix({
                "Training loss(es)": mse_losses.detach().cpu().numpy()
            })

    def reset(self):
        self.prev_sol = np.tile(0, [self.task_horizon])
        self.optimizer.reset()

    def expand_to_model_format(self, mat):
        dim = mat.shape[-1]

        # Before, [10, 5] in case of proc_obs
        reshaped = mat.view(-1, self.models.num_models, self.n_particles // self.models.num_models, dim)
        # After, [2, 5, 1, 5]

        transposed = reshaped.transpose(0, 1)
        # After, [5, 2, 1, 5]

        reshaped = transposed.contiguous().view(self.models.num_models, -1, dim)
        # After. [5, 2, 5]

        return reshaped

    def flatten_to_matrix(self, model_format):
        dim = model_format.shape[-1]

        reshaped = model_format.view(self.models.num_models, -1, self.n_particles // self.models.num_models, dim)

        transposed = reshaped.transpose(0, 1)

        reshaped = transposed.contiguous().view(-1, dim)

        return reshaped
