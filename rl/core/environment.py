from abc import ABC, abstractmethod


class Environment(ABC):
    """
    abstract class for environment
    unlike the Env class in OpenAI Gym, the agent(s) are elements of the Environment
    includes dynamics of the system, distribution of observation(s) to agent(s), auto-play (with termination rule)
    """

    @abstractmethod
    def step(self):
        """
        execute one step of dynamics
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
    def play(self):
        """
        auto-play until termination
        Returns:

        """
        pass

    @abstractmethod
    def reset(self):
        """
        resets everything to initial status
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
