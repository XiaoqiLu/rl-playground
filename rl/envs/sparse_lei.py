import numpy as np

from rl.agents import AgentDiscreteRandom, AgentDiscreteQ
from rl.core import Environment
from rl.utils import multiple_seeds, Recorder


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
    recorder = Recorder()

    def __init__(self, nuisance=0, tau=0.0, shift=-10.0, state_noise=1.0, reward_noise=1.0, max_step=10,
                 agent=None, show=True, record=True):
        self.nuisance = nuisance
        self.tau = tau
        self.shift = shift
        self.state_noise = state_noise
        self.reward_noise = reward_noise
        self.max_step = max_step
        if agent is None:
            self.agent = AgentDiscreteRandom((0, 1))
        else:
            self.agent = agent
        self.show = show
        self.record = record

        self.dim = 3 + self.nuisance
        self.rng = np.random.RandomState()

        self.A = 0.4 * np.identity(self.dim)
        if self.nuisance > 1:
            self.A[3:, 3:] -= 0.1 * np.diagflat(np.ones(self.nuisance - 1), 1)
            self.A[3:, 3:] -= 0.1 * np.diagflat(np.ones(self.nuisance - 1), -1)

        self.reset()

    def set_agent(self, agent):
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
        self.state = self.state_noise * self.rng.randn(self.dim)

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
    env = SparseLei(shift=0, reward_noise=0, max_step=100)
    env.play()
    cum_reward = env.recorder.sum(fun=lambda data: data['reward'], disc=0.9, start=1)
    print(cum_reward)

    theta = np.array([1, 1, 1, 1, 0, 0, 0, 0])


    def q(state, action):
        state_ = np.insert(state[:3], 0, 1)
        if action == 0:
            return np.dot(theta[:4], state_)
        else:
            return np.dot(theta[4:], state_)


    env = SparseLei(shift=0, reward_noise=0, max_step=10, agent=AgentDiscreteQ((0, 1), q=q))
    env.play()
    cum_reward = env.recorder.sum(fun=lambda data: data['reward'], disc=0.9, start=1)
    print(cum_reward)
