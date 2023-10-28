from jax import random
from jax import value_and_grad
from jax import jit

import jax.numpy as jnp

from modules.models.jax_backend.losses import l1, l2
from modules.models.jax_backend.activation_functions import linear
from modules.models.jax_backend.neural_networks.architectures import mlp


def fourier_seasonality(
    seasonality_init_method,
    link_function,
    loss_function,
    prngkey,
    reg_strength=0.1,
):
    """Module generating seasonality using a combination of
    sine and cosine base functions.
    """

    @jit
    def init_params(fourier_matrix):
        _, seasonality_key = random.split(prngkey)
        params = {
            "fourier_coef": seasonality_init_method(
                init_state=((fourier_matrix.shape[1],), None),
                random_key=seasonality_key,
            )
        }
        return params

    @jit
    def forward(X, current_params, random_state):
        theta = jnp.dot(X, current_params["fourier_coef"])
        yhat = link_function(theta)
        return yhat

    @jit
    def backward(X, y, current_params, random_state):
        @jit
        def compute_loss(params):
            yhat = forward(X=X, current_params=params, random_state=random_state)
            raw_loss = loss_function(y=y, yhat=yhat)
            reg_loss = l1(params=params) * reg_strength
            return raw_loss + reg_loss

        grad_func = value_and_grad(compute_loss)
        loss, grads = grad_func(current_params)
        return loss, grads

    return init_params, forward, backward


def custom_ghi_model(
    seasonality_init_method,
    covariates_init_method,
    mlp_layers,
    mlp_init_method,
    mlp_hidden_link_function,

    link_function,
    loss_function,
    prngkey,

    covariates_reg_strength=0.001,
    seasonality_reg_strength=0.01,
):
    """Custom ghi model with multiple seasonality, autoregressive component and
    covariates modelled by an MLP.
    """
    (
        yearly_seasonality_prng_key,
        daily_seasonality_prng_key,
        covariates_prng_key,
        mlp_prng_key,
    ) = random.split(prngkey, 4)

    yearly_seasonality_init_params, yearly_seasonality_forward, _ = fourier_seasonality(
        seasonality_init_method=seasonality_init_method,
        link_function=linear,
        loss_function=loss_function,
        prngkey=yearly_seasonality_prng_key,
        reg_strength=seasonality_reg_strength,
    )

    daily_seasonality_init_params, daily_seasonality_forward, _ = fourier_seasonality(
        seasonality_init_method=seasonality_init_method,
        link_function=linear,
        loss_function=loss_function,
        prngkey=daily_seasonality_prng_key,
        reg_strength=seasonality_reg_strength,
    )

    mlp_init_params, mlp_forward, _ = mlp(
        layers=mlp_layers,
        weights_init_method=mlp_init_method,
        bias_init_method=mlp_init_method,
        hidden_link_function=mlp_hidden_link_function,
        link_function=linear,
        loss_function=loss_function,
        reg_function=l2,
        prngkey=mlp_prng_key,
        reg_strength=covariates_reg_strength,
    )

    @jit
    def init_params(yearly_fourier_matrix, daily_fourier_matrix, covariates):
        params = {
            "yearly_seasonality": yearly_seasonality_init_params(
                fourier_matrix=yearly_fourier_matrix
            ),
            "daily_seasonality": daily_seasonality_init_params(
                fourier_matrix=daily_fourier_matrix
            ),
            "mlp": mlp_init_params(X_shape=covariates.shape[1], y_shape=1),
        }
        return params

    @jit
    def get_components(X, current_params, random_state):
        (
            fourier_matrix_year,
            fourier_matrix_day,
            covariates,
        ) = X

        components = {
            "yearly_seasonality": yearly_seasonality_forward(
                X=fourier_matrix_year,
                current_params=current_params["yearly_seasonality"],
                random_state=random_state,
            ),
            "daily_seasonality": daily_seasonality_forward(
                X=fourier_matrix_day,
                current_params=current_params["daily_seasonality"],
                random_state=random_state,
            ),
            "covariates": jnp.ravel(
                mlp_forward(
                    X=covariates,
                    current_params=current_params["mlp"],
                    random_state=random_state,
                )
            ),
        }

        return components

    @jit
    def forward(X, current_params, random_state):
        components = get_components(X, current_params, random_state)
        yhat = link_function(sum(components.values()))
        return yhat

    @jit
    def backward(X, y, current_params, random_state):
        @jit
        def compute_loss(params):
            yhat = forward(X=X, current_params=params, random_state=random_state)
            raw_loss = loss_function(y=y, yhat=yhat)

            yearly_seasonality_reg_loss = (
                l1(params=params["yearly_seasonality"]) * seasonality_reg_strength
            )
            daily_seasonality_reg_loss = (
                l1(params=params["daily_seasonality"]) * seasonality_reg_strength
            )

            covariates_reg_loss = l2(params=params["mlp"]) * covariates_reg_strength

            return (
                raw_loss
                + yearly_seasonality_reg_loss
                + covariates_reg_loss
                + daily_seasonality_reg_loss
            )

        grad_func = value_and_grad(compute_loss)
        loss, grads = grad_func(current_params)
        return loss, grads

    return init_params, forward, backward, get_components
