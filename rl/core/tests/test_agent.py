import numpy as np

from rl.core import agent


def test_class_agent():
    class TestAgent(agent.Agent):

        def __init__(self):
            self.action_space = (-1, 1)
            self.rng = np.random.RandomState()

        def act(self, observation):
            action = self.rng.choice(self.action_space)
            return action

        def reset(self):
            return self

        def seed(self, seed):
            self.rng.seed(seed)
            return self

    test_agent = TestAgent()
    test_agent.seed(1)
    assert test_agent.act(None) == 1
