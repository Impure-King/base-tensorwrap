"""Various Loss Functions for training the models."""

import jax
from jax import numpy as jnp
from tensorwrap.nn.models import Module
from jax import jit
from functools import partial




# @jax.tree_util.register_pytree_node_class
# class MSE(Module):
#     """Computed the square of the mean of the loss. Requires y_true and y_pred respectively."""
#     def __init__(self, y_true, y_pred):
#         super().__init__(y_true = y_true, y_pred = y_pred)
    
#     def loss_fn(y_true, y_pred):
#             return jnp.square(jnp.mean(y_pred - y_true))

#     def __call__(self): # Using call in order to make the class callable.
#         return MSE.loss_fn(self.y_true, self.y_pred)

# @jax.tree_util.register_pytree_node_class   
# class MAE(Module):
#     """Computed the square of the mean of the loss. Requires y_true and y_pred respectively."""
#     def __init__(self, y_true, y_pred):
#         super().__init__(y_true = y_true, y_pred = y_pred)
    
#     def loss_fn(y_true, y_pred):
#             return jnp.absolute(jnp.mean(y_pred - y_true))

#     def __call__(self): # Using call in order to make the class callable.
#         return MSE.loss_fn(self.y_true, self.y_pred)


def MSE(y_true, y_pred):
    return jnp.square(jnp.mean(y_pred - y_true))

def MAE(y_true, y_pred):
    return jnp.absolute(jnp.mean(y_pred - y_true))