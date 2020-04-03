import numpy as np

from rl.utils import seeding


def test_multiple_seeds():
    # common usage
    seeds = seeding.multiple_seeds(seed=1, size=4)
    assert (seeds == np.array([895547922, 2141438069, 1546885062, 2002651684])).all()
    # size = 1
    seeds = seeding.multiple_seeds(seed=1, size=1)
    assert (seeds == np.array([895547922])).all()
    # default input
    seeds = seeding.multiple_seeds()
    assert isinstance(seeds, int)
