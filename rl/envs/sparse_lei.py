import numpy as np

from rl.core.agent import Agent
from rl.core.environment import Environment
from rl.utils.seeding import multiple_seeds


class AgentRandom(Agent):

    def __init__(self, action_space, p=None):
        self.action_space = action_space
        self.p = p

        self.rng = np.random.RandomState()

    def act(self, observation):
        action = self.rng.choice(self.action_space, self.p)
        return action

    def reset(self):
        return self

    def seed(self, seed):
        self.rng.seed(seed)
        return self


class SparseLei(Environment):
    """
    the Sparse Lei environment is inspired from the paper
    An Actor-Critic Contextual Bandit Algorithm for Personalized Mobile Health Interventions
    by Lei et al
    the details of the original generative model is in session 5.3 of arXiv:1706.09090v1 [stat.ML]
    SparseLei introduces nuisance dimensions to test if RL algorithm can pick up the "sparsity" structure from data
    """

    def __init__(self, nuisance=0, tau=0.0, shift=-10.0, noise=1.0, agent=None, show=True):
        self.nuisance = nuisance
        self.tau = tau
        self.shift = shift
        self.noise = noise
        if agent is None:
            self.agent = AgentRandom((-1, 1))
        else:
            self.agent = agent
        self.show = show

        self.n_step = 0
        self.rng = np.random.RandomState()
        self.state = np.zeros(3 + self.nuisance)

        self.A = 0.4 * np.identity(3 + self.nuisance)
        if self.nuisance > 1:
            self.A[3:, 3:] -= 0.1 * np.diagflat(np.ones(self.nuisance - 1), 1)
            self.A[3:, 3:] -= 0.1 * np.diagflat(np.ones(self.nuisance - 1), -1)

    def set_agent(self, agent):
        self.agent = agent
        return self

    def step(self):
        self.n_step += 1
        action = self.agent.act(self.state)
        state_new = np.dot(self.A, self.state) + self.noise * self.rng.randn(3 + self.nuisance)
        state_new[2] += 0.2 * self.state[2] * action + 0.4 * action
        self.state = state_new
        reward = self.shift + 0.4 * (self.state[0] + self.state[1]) + \
                 0.2 * action * (1 + self.state[0] + self.state[1]) - self.tau * self.state[2] + \
                 self.noise * self.rng.randn()
        if self.show:
            print("-" * 20)
            print("action: ", action)
            print("state: ", self.state)
            print("reward: ", reward)
        return self

    def is_terminated(self):
        return self.n_step > 10

    def play(self):
        while not self.is_terminated():
            self.step()
        if self.show:
            print("-" * 20)
            print("Terminated!")
        return self

    def reset(self):
        self.n_step = 0
        self.state = np.zeros(3 + self.nuisance)
        return self

    def seed(self, seed):
        seeds = multiple_seeds(seed, 2)
        self.rng.seed(seeds[0])
        self.agent.seed(seeds[1])
        return self
