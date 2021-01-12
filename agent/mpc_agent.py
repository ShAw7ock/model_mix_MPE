import os
import numpy as np
import torch

from mpc_utils.optimizer import CEMOptimizer
from mpc_utils.dynamic_model import nn_constructor

import tqdm

TORCH_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def shuffle_rows(arr):
    # np.argsort(a, axis=-1, kind='quicksort', order=None) 按a中的元素按axis方向从小到大排序后，提取对应的索引index
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]


class MPC_Controller:
    def __init__(self, params):
        # config.pprint()
        # TODO: When using the Particle Environment, I should to consider weather to change the 'dim_o' and 'dim_u'
        self.dim_o, self.dim_a = params.env.observation_space.shape[0], params.env.action_space.shape[0]
        self.act_ub, self.act_lb = params.env.action_space.high, params.env.action_space.low
        self.per = params.per

        # Model initial config
        self.num_models = params.num_models
        self.model_input_dim = params.input_dim
        self.model_output_dim = params.output_dim
        self.model_train_epochs = params.train_epochs
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
            sol_dim=self.task_horizon * self.dim_a, max_iters=self.opt_max_iters, popsize=self.opt_popsize,
            num_elites=self.opt_num_elites, cost_function=self._compile_cost,
            lower_bound=self.act_ub,
            upper_bound=self.act_ub,
            alpha=self.opt_alpha
        )

        # Agent Controller variables
        self.has_been_trained = params.model_pretrained
        self.act_buffer = np.array([]).reshape(0, self.dim_a)
        self.prev_sol = np.tile((self.act_lb + self.act_ub) / 2, [self.task_horizon])
        self.init_var = np.tile(np.square(self.act_ub - self.act_lb) / 16, [self.task_horizon])
        # self.train_in plays the role of buffer to store (s,a)
        self.train_in = np.array([]).reshape(0, self.dim_a + self.obs_preproc(np.zeros([1, self.dim_o])).shape[-1])
        self.train_targs = np.array([]).reshape(
            0, self.targ_proc(np.zeros([1, self.dim_o]), np.zeros([1, self.dim_o])).shape[-1]
        )

        # Set up the local model for each agent
        self.models = nn_constructor(num_models=self.num_models,
                                     input_dim=self.model_input_dim,
                                     output_dim=self.model_output_dim)

    def model_transition_train(self):
        pass

    def reset(self):
        pass

    def select_action(self, obs):   # obs: The current observation
        if not self.has_been_trained:
            return np.random.uniform(self.act_lb, self.act_ub, self.act_lb.shape)
        if self.act_buffer.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action

        self.sy_cur_obs = obs

        soln = self.optimizer.obtain_solution(self.prev_sol, self.init_var)
        # self.per 决定action sequence多久更新一个，故动作序列中的动作个数为 self.per * self.dU (self.per = 1)
        self.prev_sol = np.concatenate([np.copy(soln)[self.per * self.dim_a:], np.zeros(self.per * self.dim_a)])
        self.act_buffer = soln[:self.per * self.dim_a].reshape(-1, self.dim_a)

        return self.select_action(obs)

    def _compile_cost(self, act_seqs):
        pass

    def _predict_next_obs(self):
        pass

