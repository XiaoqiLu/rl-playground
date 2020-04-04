import numpy as np


def multiple_seeds(seed=None, size=None):
    """
    generate multiple seeds from single seed
    Args:
        seed: the seed used to generate more seeds
        size: # of seeds to generate

    Returns:
        seeds

    """
    rng = np.random.RandomState(seed=seed)
    seeds = rng.tomaxint(size=size)
    return seeds
