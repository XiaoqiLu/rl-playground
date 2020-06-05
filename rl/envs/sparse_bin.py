import numpy as np

from rl.agents import AgentDiscreteRandom
from rl.core import Environment
from rl.utils import multiple_seeds


class SparseBin(Environment):
    """
    extension of Bin's toy example
    """

    n_step = 0
    action = None
    reward = None
    state = None

    def __init__(self, nuisance=0, reward_factor=1.0, reward_noise=1.0, max_step=10):
        self.nuisance = nuisance
        self.reward_factor = reward_factor
        self.reward_noise = reward_noise
        self.max_step = max_step
        self.agent = AgentDiscreteRandom((0, 1))

        self.dim = 1 + self.nuisance
        self.rng = np.random.RandomState()

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
        mu = self.state[0] * self.action
        self.reward = self.reward_factor * mu + self.reward_noise * self.rng.randn()
        state_new = 0.6 * self.state + 0.8 * self.rng.randn(self.dim)
        state_new[0] = mu + self.rng.randn()
        self.state = state_new

        meta_data = {'action': int(self.action),
                     'reward': self.reward,
                     'state': self.state.tolist()}
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
        self.state = self.rng.randn(self.dim)
        self.agent.reset()
        meta_data = {'state': self.state.tolist()}
        return meta_data

    def seed(self, seed):
        seeds = multiple_seeds(seed, 2)
        self.rng.seed(seeds[0])
        self.agent.seed(seeds[1])
        return self


if __name__ == '__main__':
    env = SparseBin()
    recorder = env.play(n=2)
    print(recorder[0].sum(lambda data: data['reward'], discount=0, start=1))
