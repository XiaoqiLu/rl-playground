from abc import ABC, abstractmethod

from rl.utils import Recorder


class Environment(ABC):
    """
    abstract class for environment
    different from the Env class in OpenAI Gym, the agent(s) are elements of the Environment
    this makes auto-play possible via play() method

    the following methods need to be implemented:
        step
        render
        is_terminated
        reset
        seed
    """

    def play(self, n=1, render=True):
        """
        auto-play until termination
        Returns:
            recorder(s), if any
        """
        recorder = []
        for i in range(n):
            recorder.append(Recorder(init_data=self.reset()))
            if render:
                self.render()
            while not self.is_terminated():
                recorder[i].rec(new_data=self.step())
                if render:
                    self.render()
        return recorder

    @abstractmethod
    def step(self):
        """
        execute one step of dynamics
        if agents are included in the environment, then one step of actions are also executed
        Returns:
            meta-data to be written into recorder(s), if any
        """
        pass

    @abstractmethod
    def render(self):
        """
        render (print, display, or other formats) current status
        Returns:

        """
        pass

    @abstractmethod
    def is_terminated(self):
        """
        checks if the system has reached stopping criteria
        Returns:
            True/False

        """
        pass

    @abstractmethod
    def reset(self):
        """
        resets everything to initial status
        note that the state of RNG(s), if any, does not reset
        Returns:
            initial meta-data to be written into recorder(s), if any
        """
        pass

    @abstractmethod
    def seed(self, seed):
        """
        resets RNG(s), if any, to initial state by seed
        Args:
            seed: seed to initialize RNG(s)

        Returns:

        """
        pass
