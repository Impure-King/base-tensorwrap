from jax import numpy as np

def expand_dims(array, axis):
    if axis==1:
        array = array.reshape(-1, 1)
    elif axis==0:
        array = array.reshape(1, -1)

    return array