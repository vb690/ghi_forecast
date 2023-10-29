from jax import random
from jax import value_and_grad
from jax import jit

from tqdm import tqdm

from jax.tree_util import tree_map, tree_leaves

import jax.numpy as jnp


def newton_rhapson(learning_rate, beta=0.9):
    """
    Implementation of newton-rhapson algorithm for SGD,
    the algorithm supports momentum.
    """

    @jit
    def init_state(params):
        previous_updates = tree_map(lambda param: jnp.zeros_like(param), params)
        return previous_updates, params

    @jit
    def update_state(grads, current_state):
        previous_updates, previous_params = current_state
        current_updates = tree_map(
            lambda grad, previous_update: beta * previous_update
            + (jnp.clip(grad, -1, 1) * learning_rate),
            grads,
            previous_updates,
        )
        current_params = tree_map(
            lambda param, update: param - update, previous_params, current_updates
        )
        return current_updates, current_params

    @jit
    def get_params(state):
        _, current_params = state
        return current_params

    return init_state, update_state, get_params
