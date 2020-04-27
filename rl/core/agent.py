from abc import ABC, abstractmethod


class Agent(ABC):
    """
    abstract class for agent

    the following methods need to be implemented
        act
        reset
        seed
    """

    @abstractmethod
    def act(self, obs):
        """
        takes observation and outputs action
        Args:
            obs: observation feeding to agent

        Returns:
            action

        """
        pass

    @abstractmethod
    def reset(self):
        """
        resets agent status to initial state
        note that the state of RNG(s), if any, does not reset
        Returns:

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
