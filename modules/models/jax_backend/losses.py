from jax import jit

from jax.tree_util import tree_leaves

import jax.numpy as jnp


@jit
def mae(y, yhat):
    """Mean absolute error"""
    error = jnp.abs(y - yhat)
    return jnp.mean(error)


@jit
def l1(params):
    """L1 norm applied to the parameters"""
    l1_loss = sum([jnp.sum(jnp.abs(leave)) for leave in tree_leaves(params)])
    return l1_loss


@jit
def l2(params):
    """L2 norm applied to the parameters"""
    l2_loss = sum([jnp.sum(jnp.square(leave)) for leave in tree_leaves(params)])
    return l2_loss
