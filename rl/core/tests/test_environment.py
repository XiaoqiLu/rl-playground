import numpy as np

from rl.core.agent import Agent
from rl.core.environment import Environment
from rl.utils.seeding import multiple_seeds


def test_class_environment():
    class TestAgent(Agent):

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

    class TestEnvironment(Environment):

        def __init__(self, agent: Agent, show=True):
            self.agent = agent
            self.show = show

            self.rng = np.random.RandomState()
            self.n_step = 0
            self.state = 0
            self.reset()

        def step(self):
            self.n_step += 1
            observation = self.state
            action = self.agent.act(observation)
            self.state += action + self.rng.randn()
            if self.show:
                print("-" * 20)
                print("step " + str(self.n_step))
                print("observation " + str(observation))
                print("action " + str(action))
            return self

        def is_terminated(self):
            return np.abs(self.state) > 10

        def play(self):
            while not self.is_terminated():
                self.step()
            if self.show:
                print("-" * 20)
                print("Terminated!")
            return self

        def reset(self):
            self.n_step = 0
            self.state = 0
            return self

        def seed(self, seed):
            seeds = multiple_seeds(seed, 2)
            self.rng.seed(seeds[0])
            self.agent.seed(seeds[1])
            return self

    print("")
    test_agent = TestAgent()

    test_environment = TestEnvironment(agent=test_agent)
    test_environment.seed(1)
    test_environment.play()
    assert test_environment.n_step == 24
    test_environment.reset()
    test_environment.play()
    assert test_environment.n_step == 43
