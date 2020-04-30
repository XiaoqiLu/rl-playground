import numpy as np
from scipy.special import softmax

from rl.core import Agent


class AgentDiscreteRandom(Agent):

    def __init__(self, action_space, p=None):
        self.action_space = action_space
        self.p = p

        self.rng = np.random.RandomState()

    def act(self, obs):
        action = self.rng.choice(self.action_space, p=self.p)
        return action

    def reset(self):
        return self

    def seed(self, seed):
        self.rng.seed(seed)
        return self


class AgentDiscreteQ(Agent):

    def __init__(self, action_space, q, dist='max', temperature=1.0):
        self.action_space = action_space
        self.q = q
        if dist in ('max', 'gibbs'):
            self.dist = dist
        else:
            self.dist = 'uniform'
            print("Method not specified, using uniform distribution.")
        self.temperature = temperature

        self.rng = np.random.RandomState()

    def act(self, obs):
        values = np.array(list(map(lambda a: self.q(obs, a), self.action_space)))
        if self.dist == 'max':
            action = np.argmax(values)
        elif self.dist == 'gibbs':
            action = self.rng.choice(self.action_space, p=softmax(values / self.temperature))
        else:
            action = self.rng.choice(self.action_space)
        return action

    def reset(self):
        return self

    def seed(self, seed):
        self.rng.seed(seed)
        return self
