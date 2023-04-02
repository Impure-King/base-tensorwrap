from jax import (numpy as np,
                 jit,
                 Array)

@jit
def last_dim(array: Array):
    r"""Returns the last dimension of the array, list, or integer. Used internally for Dense Layers and Compilations.
    
    Arguments:
        array (Array): Array for size computation
    """
    try:
        return np.shape(array)[-1]
    except:
        return array

def comprehend(hist, type: str):
    """A method that returns the loss/metric w.r.t. epoch from the return value of .fit"""
    dic = {
        "metric" : 1,
        "loss" : 0
    }
    value = [a[dic[type]] for a in list(hist.values())]
    return list(hist.keys()), value

    