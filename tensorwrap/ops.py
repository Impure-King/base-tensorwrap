from jax import (numpy as np,
                 jit,)

def expand_dims(array, axis):
    if axis==1:
        array = array.reshape(-1, 1)
    elif axis==0:
        array = array.reshape(1, -1)

    return array

@jit
def shape(array):
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    try:
        return array.shape[-1]
    except:
        return array
    
    