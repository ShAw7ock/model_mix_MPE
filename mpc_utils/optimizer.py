import numpy as np
import scipy.stats as stats


class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def reset(self):
        raise NotImplementedError("Must be implemented in subclass.")

    def obtain_solution(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")


class CEMOptimizer(Optimizer):
    def __init__(self, agent_num, sol_dim, action_space, max_iters, popsize, num_elites,
                 epsilon=0.001, alpha=0.25):
        """
        Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space (task_horizon * action_dim)
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super(CEMOptimizer, self).__init__()
        self.agent_num = agent_num
        self.action_space = action_space
        self.sol_dim = sol_dim
        self.max_iters = max_iters
        self.popsize = popsize
        self.num_elites = num_elites

        self.epsilon, self.alpha = epsilon, alpha

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

    def reset(self):
        pass

    # 暂时使用Random shooting的方法
    def obtain_solution(self, rin_function):
        solutions = np.random.randint(0, self.action_space, size=[self.popsize, self.sol_dim])
        returns, hidden_state_candidate = rin_function(solutions, self.agent_num)
        return solutions[np.argmax(returns)], hidden_state_candidate[np.argmax(returns)]

    def prob_update_solution(self, prev_prob, rin_function):
        probs, t = prev_prob, 0
        act_candidate = np.arange(self.action_space)

        X = stats.rv_discrete(values=(act_candidate, probs))

        samples = X.rvs(size=[self.popsize, self.sol_dim])

        returns = rin_function(samples)
        elites = samples[np.argsort(returns)][-self.num_elites:]

        nums_act, this_probs = np.zeros_like(act_candidate), np.zeros_like(act_candidate)
        for i in range(elites.shape[0]):
            for j in range(elites.shape[1]):
                cur = elites[i][j]
                nums_act[cur] += 1
        for i in range(self.action_space):
            this_probs[i] = nums_act[i] / (self.num_elites * self.sol_dim)

        return samples[np.argmax(returns)], this_probs
