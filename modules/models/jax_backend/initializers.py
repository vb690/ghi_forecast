from jax import random


def random_gaussian(init_state, random_key, sigma: float = 1):
    """Initialize with random values sampled from a gaussian distribution with mean
    zero and standard deviation of sigma.
    """
    params_shape, _ = init_state
    params = random.normal(key=random_key, shape=params_shape) * sigma
    return params
