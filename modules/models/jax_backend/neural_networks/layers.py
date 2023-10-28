from functools import partial

import jax.numpy as jnp
from jax import jit
from jax import random


def dense(weights_init_method, bias_init_method):
    """Implementation of a dense layer"""

    @partial(jit, static_argnums=(0, 1))
    def init_params(input_shape, hidden_units, prngkey):
        weights_key, bias_key = random.split(prngkey)
        params = {
            "weights": weights_init_method(
                init_state=((input_shape, hidden_units), input_shape),
                random_key=weights_key,
            ),
            "bias": bias_init_method(
                init_state=((hidden_units,), None), random_key=bias_key
            ),
        }
        return params

    @jit
    def forward(X, current_params, random_state):
        theta = jnp.dot(X, current_params["weights"]) + current_params["bias"]
        return theta

    return init_params, forward
