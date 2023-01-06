import jax.numpy as jnp
from jax import jit
from functools import partial



@partial(jit, static_argnums = 1)
def expand_dims(tensor, axis = None, dtype = 'float32'):
    """Used to add an extra dimension at the end of the tensor, along different axis."""
    if tensor == None:
        raise ValueError("Input Tensor missing. Enter a tensor to apply this operation on.")
    if axis == 1:
        array = jnp.reshape(tensor, (-1, 1))
    elif axis == 0 or axis == -1:
        array = jnp.reshape(tensor, (1, -1))
    else:
        raise ValueError("Axis input is incorrect. Must be one of the following integers: -1, 0, or 1.")
    array = jnp.array(array, dtype = dtype)
    return array


def range(start, end, delta = 0, dtype = 'float32'):
    """Creates a range of values in an array.
    Start: the value to start
    End: the value to end before.
    Delta: the common difference of the sequence"""
    array = jnp.arange(start, end, step = delta, dtype= dtype)
    return array



