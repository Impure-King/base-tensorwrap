from jax import (numpy as np,
                 jit,
                 Array)

@jit
def shape(array: Array):
    r"""Returns the last dimension of the array, list, or integer. Used internally for Dense Layers and Compilations.
    
    Arguments:
        array (Array): Array for size computation
    """
    try:
        return np.shape(array)[-1]
    except:
        return array


    