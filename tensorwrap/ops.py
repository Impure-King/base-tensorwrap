import jax
import random
from jax import (numpy as np,
                 jit,
                 Array)
from typing import (Tuple,
                    Optional,
                    Any)


def last_dim(array: Array):
    r"""Returns the last dimension of the array, list, or integer. Used internally for Dense Layers and Compilations.
    
    Arguments:
        array (Array): Array for size computation
    """
    try:
        return np.shape(array)[-1]
    except:
        return array


def randu(shape: Tuple[int, ...], key = jax.random.PRNGKey(random.randint(1, 5))):
    """Returns a uniformly distributed random tensor.
    
    Arguments:
        - shape: A tuple containing the dimensions of the return tensor.
        - key (Optional): A jax.random.PRNGKey that determines the reproducibility of the tensor."""
    return jax.random.uniform(key, shape, dtype=np.float32)


def randn(shape: Tuple[int, ...], key = jax.random.PRNGKey(random.randint(1, 5))):
    """Returns a normally distributed random tensor.
    
    Arguments:
        - shape: A tuple containing the dimensions of the return tensor.
        - key (Optional): A jax.random.PRNGKey that determines the reproducibility of the tensor."""
    return jax.random.normal(key, shape, dtype=np.float32)