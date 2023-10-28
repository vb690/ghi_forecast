from jax import random
from jax import value_and_grad
from jax import jit

from tqdm import tqdm

from jax.tree_util import tree_map, tree_leaves

import jax.numpy as jnp


def train_step(X, y, backward, current_state, update_state, get_params, random_state):
    current_params = get_params(state=current_state)
    loss, grads = backward(
        X=X, y=y, current_params=current_params, random_state=random_state
    )
    current_state = update_state(grads, current_state)

    return loss, current_state, grads


def fit(
    X,
    y,
    backward,
    start_params,
    optimizer,
    epochs,
    stopper,
    verbose,
    random_key=666,
    batch_size=None,
    track_grads=False,
):
    history = []
    grads_history = []
    random_state = random.PRNGKey(random_key)

    init_state, update_state, get_params = optimizer
    current_state = init_state(params=start_params)
    pbar = tqdm(range(epochs), desc="Loss")
    for epoch in pbar:
        random_state, _ = random.split(random_state)

        if batch_size is None:
            loss, current_state, grads = train_step(
                X=X,
                y=y,
                backward=backward,
                current_state=current_state,
                update_state=update_state,
                get_params=get_params,
                random_state=random_state,
            )

        history.append(loss)
        if track_grads:
            grads_history.append(grads)
        if epoch % verbose == 0:
            pbar.set_description(f"Loss {jnp.round(loss, 4)}")
        if stopper.evaluate(current_loss=loss):
            break

    if track_grads:
        return get_params(state=current_state), history, grads_history
    else:
        return get_params(state=current_state), history
