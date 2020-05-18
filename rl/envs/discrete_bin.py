import numpy as np

from rl.agents import AgentDiscreteRandom
from rl.core import Environment
from rl.utils import multiple_seeds, Recorder


class DiscreteBin(Environment):
    """
    Discrete Bin(ary) environment to confirm Bin's conjecture about edges
    """

    n_step = 0
    action = None
    reward = None
    state = None
    recorder = Recorder()

    def __init__(self, state_noise=0.5, reward_noise=0.0, max_step=10, agent=None, show=True, record=True):
        self.state_noise = state_noise
        self.reward_noise = reward_noise
        self.max_step = max_step
        if agent is None:
            self.agent = AgentDiscreteRandom((0, 1))
        else:
            self.agent = agent
        self.show = show
        self.record = record

        self.rng = np.random.RandomState()

        self.reset()

    def set_agent(self, agent):
        self.agent = agent
        return self

    def step(self):
        self.n_step += 1
        self.action = self.agent.act(self.state)
        self.reward = self.state[0] - self.state[1] + (2 * self.state[2] - 1) * self.action + \
                      self.reward_noise * self.rng.randn(1)
        mu = 0.5 * self.state_noise + (1 - self.state_noise) * (self.state ^ self.action)
        self.state = 1 * (self.rng.rand(3) < mu)

        if self.record:
            self.recorder.rec({'action': int(self.action),
                               'reward': self.reward,
                               'state': self.state.tolist()})

        if self.show:
            print("step: ", self.n_step)
            print("  action: ", self.action)
            print("  reward: ", self.reward)
            print("  state: ", self.state)
        return self

    def is_terminated(self):
        return self.n_step >= self.max_step

    def reset(self):
        self.n_step = 0
        self.state = self.rng.randint(2, size=3)

        if self.record:
            self.recorder.reset({'state': self.state.tolist()})
        self.agent.reset()

        if self.show:
            print("  step: ", self.n_step)
            print("  state: ", self.state)
        return self

    def seed(self, seed):
        seeds = multiple_seeds(seed, 2)
        self.rng.seed(seeds[0])
        self.agent.seed(seeds[1])
        return self


if __name__ == '__main__':
    env = DiscreteBin(state_noise=0.5, max_step=100)
    env.play()
    cum_reward = env.recorder.sum(fun=lambda data: data['reward'], disc=0.9, start=1)
    print(cum_reward)
