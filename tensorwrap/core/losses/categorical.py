import tensorwrap as tf
import jax
from jax import numpy as jnp

__all__ = ["_SparseCategoricalCrossentropy"]

@jax.jit
def _SparseCategoricalCrossentropy(y_true, y_pred):
    y_true = jnp.asarray(y_true, dtype=jnp.int32)
    log_probs = jnp.log(y_pred)
    one_hot_true = jnp.eye(y_pred.shape[-1])[y_true]
    cross_entropy = jnp.sum(one_hot_true * log_probs, axis=-1)
    return -jnp.mean(cross_entropy)