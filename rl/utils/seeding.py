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


if __name__ == '__main__':
    print("-test: single seed")
    seeds = multiple_seeds(seed=0, size=1)
    print(seeds)
    rng = np.random.RandomState(seed=seeds)
    print(rng.rand())

    print("-test: multiple seeds")
    seeds = multiple_seeds(seed=0, size=5)
    print(seeds)
    rng = np.random.RandomState(seed=seeds[0])
    print(rng.rand())
