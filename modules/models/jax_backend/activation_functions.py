from jax import jit


import jax.numpy as jnp


@jit
def linear(x):
    """Activation function returning the identity of x"""
    return x


@jit
def relu(x):
    """Activation function returning the maximum between 0 and x (range 0, inf)"""
    return jnp.maximum(0, x)
