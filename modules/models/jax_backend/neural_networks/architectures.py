from functools import partial

from jax import value_and_grad
from jax import jit
from jax import random
from jax.random import KeyArray

from modules.models.jax_backend.neural_networks.layers import dense


def mlp(
    layers,
    weights_init_method,
    bias_init_method,
    hidden_link_function,
    link_function,
    loss_function,
    reg_function,
    prngkey,
    reg_strength: 0.01,
):
    """Implementation of a multilayer perceptron
    """
    dense_init_params, dense_forward = dense(
        weights_init_method=weights_init_method,
        bias_init_method=bias_init_method,
    )

    @partial(jit, static_argnums=(0, 1))
    def init_params(
        X_shape, y_shape
    ):
        seed_key, layer_key = random.split(prngkey)
        input_shape = X_shape

        params = {}
        for layer_index, hidden_units in enumerate(layers):
            params[layer_index] = dense_init_params(
                input_shape=input_shape, hidden_units=hidden_units, prngkey=layer_key
            )
            input_shape = hidden_units
            seed_key, layer_key = random.split(seed_key)

        params[len(layers)] = dense_init_params(
            input_shape=input_shape, hidden_units=y_shape, prngkey=layer_key
        )

        return params

    @jit
    def forward(X, current_params, random_state):
        theta = X
        for layer in range(len(layers)):
            theta = dense_forward(
                X=theta, current_params=current_params[layer], random_state=random_state
            )
            theta = hidden_link_function(theta)

        theta = dense_forward(
            X=theta,
            current_params=current_params[len(layers)],
            random_state=random_state,
        )
        yhat = link_function(theta)
        return yhat

    @jit
    def backward(
        X,
        y,
        current_params,
        random_state,
    ):

        @jit
        def compute_loss(params):
            yhat = forward(X=X, current_params=params, random_state=random_state)
            raw_loss = loss_function(y=y, yhat=yhat)
            reg_loss = reg_function(params=current_params) * reg_strength
            return raw_loss + reg_loss

        grad_func = value_and_grad(compute_loss)
        loss, grads = grad_func(current_params)
        return loss, grads

    return init_params, forward, backward