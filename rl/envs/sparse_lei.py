import numpy as np

from rl.agents import AgentDiscreteRandom
from rl.core import Environment
from rl.utils import multiple_seeds


class SparseLei(Environment):
    """
    the Sparse Lei environment is inspired from the paper
    An Actor-Critic Contextual Bandit Algorithm for Personalized Mobile Health Interventions
    by Lei et al
    the details of the original generative model is in session 5.3 of arXiv:1706.09090v1 [stat.ML]
    SparseLei introduces nuisance dimensions to test if RL algorithm can pick up the "sparsity" structure from data
    """

    n_step = 0
    action = None
    reward = None
    state = None

    def __init__(self, nuisance=0, tau=0.0, shift=0.0, state_noise=1.0, reward_noise=1.0, max_step=10):
        self.nuisance = nuisance
        self.tau = tau
        self.shift = shift
        self.state_noise = state_noise
        self.reward_noise = reward_noise
        self.max_step = max_step
        self.agent = AgentDiscreteRandom((0, 1))

        self.dim = 3 + self.nuisance
        self.rng = np.random.RandomState()

        self.A = 0.4 * np.identity(self.dim)
        if self.nuisance > 1:
            self.A[3:, 3:] -= 0.1 * np.diagflat(np.ones(self.nuisance - 1), 1)
            self.A[3:, 3:] -= 0.1 * np.diagflat(np.ones(self.nuisance - 1), -1)

        self.reset()

    def set_agent(self, agent=None):
        if agent is None:
            self.agent = AgentDiscreteRandom((0, 1))
        else:
            self.agent = agent
        return self

    def step(self):
        self.n_step += 1
        self.action = self.agent.act(self.state)
        self.reward = self.shift + 0.4 * (self.state[0] + self.state[1]) + \
                      0.2 * self.action * (1 + self.state[0] + self.state[1]) - self.tau * self.state[2] + \
                      self.reward_noise * self.rng.randn()
        state_new = np.dot(self.A, self.state) + self.state_noise * self.rng.randn(self.dim)
        state_new[2] += 0.2 * self.state[2] * self.action + 0.4 * self.action
        self.state = state_new

        meta_data = {'action': self.action,
                     'reward': self.reward,
                     'state': self.state}
        return meta_data

    def render(self):
        print("-- step " + str(self.n_step) + " --")
        if self.n_step == 0:
            print("state: " + str(self.state))
        else:
            print("action: " + str(self.action))
            print("reward: " + str(self.reward))
            print("state: " + str(self.state))
        return self

    def is_terminated(self):
        return self.n_step >= self.max_step

    def reset(self):
        self.n_step = 0
        self.action = None
        self.reward = None
        self.state = self.state_noise * self.rng.randn(self.dim)
        self.agent.reset()
        meta_data = {'state': self.state}
        return meta_data

    def seed(self, seed):
        seeds = multiple_seeds(seed, 2)
        self.rng.seed(seeds[0])
        self.agent.seed(seeds[1])
        return self


if __name__ == '__main__':
    env = SparseLei()
    recorders = env.play(n=2)
    print(recorders[0].sum(lambda data: data['reward'], discount=0, start=1))
