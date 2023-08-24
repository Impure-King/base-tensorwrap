# Stable Modules:
import jax
import optax
from jax import numpy as jnp
from functools import partial

__all__ = ["trainable_loop"]

@jax.value_and_grad
@partial(jax.jit, static_argnums=1)
def grad_fn(params, loss_fn, model, inputs, labels):
    y_pred = model(params, inputs)
    loss = loss_fn(labels, y_pred)
    return loss

def update(model, loss_fn, optimizer, state, inputs, labels):
    params = model.trainable_variables
    loss, grads = grad_fn(params, loss_fn, model, inputs, labels)
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return loss, params, state

def training_loop(model, loss_fn, optimizer, inputs, labels, epochs):
    state = optimizer.init(model.trainable_variables)
    for epoch in range(1, epochs+1):
        loss, model.trainable_variables, state = update(model, loss_fn, optimizer, state, inputs, labels)
        print(f"Epoch {epoch} \t\t\t {loss=}")


# Inspection Fixes:
training_loop.__module__ = "tensorwrap.nn.models"