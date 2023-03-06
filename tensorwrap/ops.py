from jax import (numpy as np,
                 jit,)

@jit
def shape(array):
    try:
        return np.shape(array)[-1]
    except:
        return array
    
    